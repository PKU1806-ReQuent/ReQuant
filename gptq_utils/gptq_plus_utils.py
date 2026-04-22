import copy
import logging
import os
import math
import pprint
import functools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import quant_utils, memory_utils, model_utils


class GPTQPlus:
    def __init__(self, 
        layer, 
        saliency: torch.Tensor, # shape (N, seq_len, G)
        gradient: torch.Tensor, # shape (G, in_features)
        num_groups: int,
        alpha: float,
        kl_loss: float,
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        # Instead of passing H in, we allocate a 3D Hessian buffer
        # that will hold sub-channel Hessians along the last dim.
        self.num_groups = num_groups
        assert self.num_groups == saliency.shape[2], "Number of groups for GuidedQuant must match saliency shape!"
        self.alpha = alpha
        self.kl_loss = max(kl_loss, 0)

        self.saliencies = saliency.float()
        self.gradients = gradient.float()
        self.H = torch.zeros(
            (self.columns, self.columns, self.num_groups),
            device=self.dev
        )
        self.act_square = torch.zeros(
            (self.columns), device=self.dev
        )
        self.nsamples = saliency.shape[0]
        self.index = 0

        # Assert row partition is valid:
        # we do the same partition as before:
        assert self.rows % self.num_groups == 0, (
            f"Number of rows ({self.rows}) must be divisible "
            f"by num_groups ({self.num_groups})"
        )

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor, out):
        """
        inp: shape [batch_size, seq_len, in_features]

        We'll slice self.saliencies[index: index + batch_size]
        do the einsum => accumulate into self.H
        then index += batch_size.
        """
        # If input is 2D or 1D, reshape to [batch, seq_len, dim] for consistency
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)  # => [1, seq_len, dim]
        else:
            assert inp.dim() == 3, "Input must be 2D or 3D. Got %dD." % inp.dim()

        bsz = inp.shape[0]
        # slice out shape => (bsz, seq_len, G)
        sal_batch = self.saliencies[self.index: self.index + bsz].to(self.dev)
        self.H *= self.index / (self.index + bsz)
        self.index += bsz

        # Flatten
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
            sal_batch = sal_batch.reshape(-1, sal_batch.shape[-1])
            
        inp = inp.float()
        sal_batch = sal_batch.float()
        n_tokens = inp.shape[0]

        sal_weighted_inp = torch.einsum("nj,ng->njg", inp, sal_batch)
        block = torch.einsum("ni,njg->ijg", inp, sal_weighted_inp)
        self.H.add_(block, alpha=1 / (n_tokens * self.index))
        self.act_square.add_((inp ** 2).sum(0), alpha=1 / n_tokens)

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        enable_gradient_update=True,
        export_to_et=False,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()
        # Prepare final Q, W_int, Scale
        Q_final = torch.zeros_like(W)
        W_int_final = torch.zeros_like(W)
        Scale_final = torch.zeros_like(W)

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        # We will partition the rows into num_groups slices
        rows_per_sub = self.rows // self.num_groups

        # Loop over each row partition, using H[..., sub_idx]
        for sub_idx in range(self.num_groups):
            row_start = sub_idx * rows_per_sub
            row_end = (sub_idx + 1) * rows_per_sub

            # Sub-slice of W
            W_sub = W[row_start:row_end, :]

            # Hessian sub-part
            H_sub = self.H[:, :, sub_idx]

            # Gradient sub-part
            gradients_sub = self.gradients[row_start: row_end, :].to(self.dev)

            # Apply the same "dead columns" logic
            dead = torch.diag(H_sub) == 0
            H_sub[dead, dead] = 1
            W_sub[:, dead] = 0

            # Static grouping for columns
            if static_groups:
                groups = []
                for i in range(0, self.columns, groupsize):
                    quantizer = copy.deepcopy(self.quantizer)
                    quantizer.find_params(W[:, i : (i + groupsize)])
                    groups.append(quantizer)

            # Possibly reorder columns by diag(H_sub)
            if actorder:
                # perm = torch.argsort(torch.diag(H_sub), descending=True)
                perm = torch.argsort(self.act_square, descending=True)
                W_sub = W_sub[:, perm]
                H_sub = H_sub[perm][:, perm]
                gradients_sub = gradients_sub[:, perm]
                invperm = torch.argsort(perm)

            # Create local buffers
            Losses = torch.zeros_like(W_sub)
            Q = torch.zeros_like(W_sub)

            damp_percent = percdamp
            damp_auto_increment = 0.0015
            while 1 > damp_percent > 0:
                try:
                    damp = damp_percent * torch.mean(torch.diag(H_sub))
                    diag = torch.arange(self.columns, device=self.dev)
                    H_sub[diag, diag] += damp
                    H_sub = torch.linalg.cholesky(H_sub)
                    H_sub = torch.cholesky_inverse(H_sub)
                    Hinv_init = H_sub
                    H_sub = torch.linalg.cholesky(H_sub, upper=True)
                    Hinv = H_sub
                    break
                except torch._C._LinAlgError as e:
                    logging.warning(f"Quantization: Current `damp_percent = {damp_percent:.5f}` is too low, auto-incrementing by `{damp_auto_increment:.5f}`")
                    damp_percent += damp_auto_increment

            if not (0 < damp_percent < 1):
                raise ValueError(f"Quantization: `damp_percent` must between 0 and 1. current is {damp_percent}")
            
            # Dynamic scaling for gradients
            if enable_gradient_update and self.alpha > 0:
                alpha = self.alpha / (self.rows * self.columns)
                GHinv = gradients_sub.matmul(Hinv_init)
                # c = (gradients_sub * GHinv).sum(dim=1) - (GHinv[:, 0] ** 2) / Hinv_init[0, 0]
                c = (gradients_sub * GHinv).sum(dim=1) - ((GHinv ** 2) / torch.diagonal(Hinv_init).unsqueeze(0)).mean(1)   # iterate over t, then take average
                # if not (c > 0).all():
                #     logging.warning(f"c in dynamic scaling contains negative values (fraction: {(c > 0).sum() / c.shape[0]}")
                c = torch.clamp(c, min=2 * alpha * self.kl_loss)
                beta = 1 - torch.sqrt(torch.clamp(1 - (2 * alpha * self.kl_loss) / c, min=0.0))
            else:
                beta = torch.zeros([1]).to(gradients_sub)

            Z = gradients_sub.matmul(Hinv.T) * beta.unsqueeze(1)
            GHinv = Z.matmul(Hinv)
            D = torch.arange(blocksize - 1, -1, -1).to(GHinv)

            W_int_sub = torch.zeros_like(W_sub)
            Scale_sub = torch.zeros_like(W_sub)

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W_sub[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                W_int1 = torch.zeros_like(W1)
                Scale1 = torch.zeros_like(W1).to(Scale_sub.dtype)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]
                GHinv1 = GHinv[:, i1:i2].clone()
                Z1 = Z[:, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(
                                    W[:, (i1 + i) : (i1 + i + groupsize)]
                                )
                        else:
                            idx = i1 + i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // groupsize]

                    q, int_weight, scale = self.quantizer.fake_quantize(
                        w.unsqueeze(1), st_idx=row_start, end_idx=row_end
                    )
                    Q1[:, i] = q.flatten()
                    q = q.flatten()
                    W_int1[:, i] = int_weight.flatten()
                    Scale1[:, i] = scale.flatten()

                    Losses1[:, i] = (w - q - GHinv1[:, i]) ** 2 / d**2

                    err1 = (w - q - GHinv1[:, i]) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) + GHinv1[:, i:]
                    Err1[:, i] = err1

                    GHinv1[:, i:] = GHinv1[:, i:] - Z1[:, i].unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))

                Q[:, i1:i2] = Q1
                W_int_sub[:, i1:i2] = W_int1
                Scale_sub[:, i1:i2] = Scale1
                Losses[:, i1:i2] = Losses1 / 2

                # Propagate error across the rest
                G_Update = blocksize * GHinv[:, i2:] - torch.einsum("ij,j,jk->ik", Z1, D, Hinv[i1:i2, i2:])
                W_sub[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) + G_Update

                # Update GHinv for the rest
                GHinv[:, i2:] -= Z[:, i1:i2].matmul(Hinv[i1:i2, i2:])

            # If we permuted columns, we un-permute the result
            if actorder:
                Q = Q[:, invperm]
                W_int_sub = W_int_sub[:, invperm]
                Scale_sub = Scale_sub[:, invperm]

            # Write the subchannel results back
            Q_final[row_start:row_end, :] = Q
            W_int_final[row_start:row_end, :] = W_int_sub
            Scale_final[row_start:row_end, :] = Scale_sub

        torch.cuda.synchronize()

        if export_to_et:
            self.layer.register_buffer(
                "int_weight", W_int_final.reshape(self.layer.weight.shape)
            )
            self.layer.register_buffer("scale", Scale_final)
        self.layer.weight.data = Q_final.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in weights")

            pprint.pprint(
                {
                    "bits": self.quantizer.bits,
                    "scale": self.quantizer.scale,
                    "zero": self.quantizer.zero,
                }
            )
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        self.Losses = None
        torch.cuda.empty_cache()
        memory_utils.cleanup_memory(verbos=False)


class SaliencyCache:
    """
    class for saving the output activation gradients in each layer.
    """
    def __init__(self, names, num_groups):
        self.num_groups = num_groups
        self.saliency_cache = {}
        self.names = names
        for name in self.names:
            self.saliency_cache[name] = []
        self.handles = []
        self.hooks_enabled = False

    def cache_saliency(self, module, inp, out, name):
        # We'll store gradient on 'out', so we must retain it
        out.retain_grad()

        def grad_hook(grad):
            """
            grad shape typically [bsz, seq_len, hidden_dim].
            We group the channels, take abs, then average.
            """
            if not self.hooks_enabled:
                return
            bsz, seq_len, hidden_dim = grad.shape
            group_size = hidden_dim // self.num_groups

            grad_squared = grad.float().pow(2).view(bsz, seq_len, self.num_groups, group_size)
            mean_squared_grad = grad_squared.mean(dim=-1)  # -> [bsz, seq_len, num_groups]

            self.saliency_cache[name].append(mean_squared_grad)

        # Attach the gradient hook to 'out'
        out.register_hook(grad_hook)

    def add_hook(self, full, enable=True):
        for name in self.names:
            self.handles.append(
                full.get(name, full.get(name + ".module", None)).register_forward_hook(
                    functools.partial(self.cache_saliency, name=name)
                )
            )
        self.hooks_enabled = enable

    def enable_hooks(self):
        self.hooks_enabled = True

    def disable_hooks(self):
        self.hooks_enabled = False

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.hooks_enabled = False
        memory_utils.cleanup_memory()

    def clear_cache(self):
        for name in self.names:
            self.saliency_cache[name] = []
        memory_utils.cleanup_memory()


class GradientCache:
    """
    class for saving the weight gradients in each layer.
    """
    def __init__(self, names, num_groups):
        self.num_groups = num_groups
        self.gradients_cache = {}
        self.index = {}
        self.names = names
        for name in self.names:
            self.gradients_cache[name] = 0
            self.index[name] = 0
        self.handles = []
        self.hooks_enabled = False

    def cache_gradient(self, grad, name):
        if not self.hooks_enabled:
            return
        self.gradients_cache[name] *= self.index[name] / (self.index[name] + 1)
        self.index[name] += 1
        self.gradients_cache[name] += grad.float() / self.index[name]

    def add_hook(self, full, enable=True):
        for name in self.names:
            self.handles.append(
                full.get(name, full.get(name + ".module", None)).weight.register_hook(
                    functools.partial(self.cache_gradient, name=name)
                )
            )
        self.hooks_enabled = enable

    def enable_hooks(self):
        self.hooks_enabled = True

    def disable_hooks(self):
        self.hooks_enabled = False

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.hooks_enabled = False
        memory_utils.cleanup_memory()

    def clear_cache(self):
        for name in self.names:
            self.gradients_cache[name] = 0
            self.index[name] = 0
        memory_utils.cleanup_memory()


def hidden2logits(hidden_states, analyzer: model_utils.ModelAnalyzer):
    norm = analyzer.get_layernorm_before_head()
    lm_head = analyzer.get_lm_head()

    logits = lm_head(norm(hidden_states))

    return logits


@torch.no_grad()
def gptq_fwrd(args, analyzer: model_utils.ModelAnalyzer, dataloader, dev):
    """
    From GPTQ repo
    """
    logging.info("-----GPTQPlus Quantization-----")

    model = analyzer.model
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = analyzer.get_layers()
    orig_device = next(model.parameters()).device

    for module in analyzer.get_pre_block_modules() + [
        analyzer.get_layernorm_before_head(),
        analyzer.get_lm_head(),
    ]:
        module.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].to(orig_device)
    memory_utils.cleanup_memory(False)

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    sequential = analyzer.get_sequential_quantizable_module_names()
    names = [n for ns in sequential for n in ns]
    fp_inps = inps.clone()

    if args.offload_inps:
        inps = inps.cpu()
        fp_inps = fp_inps.cpu()

    quantizers = {}
    pbar = tqdm(range(len(layers)), ncols=120, desc="Quantizing Layers", position=0)
    for i in pbar:
        layer = layers[i].to(dev)
        full = analyzer.get_quantizable_modules(layer)

        bits_config = quant_utils.disable_act_quant(layer)
        for j in range(args.nsamples):
            fp_inps[j] = layer(fp_inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask, position_ids=position_ids,
                               position_embeddings=position_embeddings)[0].to(fp_inps.device)
        quant_utils.enable_act_quant(layer, bits_config)

        #############################################################################

        # Prepare for batched inference (TODO. Currently assume equal length in a batch, support variable length?)
        batch_attention_mask = attention_mask.expand(args.bsz, -1, -1, -1)
        batch_position_ids = position_ids.expand(args.bsz, -1)
        batch_position_embeddings = (
            position_embeddings[0].expand(args.bsz, -1, -1),
            position_embeddings[1].expand(args.bsz, -1, -1)
        )

        # Get per-block gradients and hessians
        with torch.enable_grad():
            # Add forward hooks
            saliency_cache = SaliencyCache(names, args.num_groups)
            saliency_cache.add_hook(full, enable=False)
            gradients_cache = GradientCache(names, args.num_groups)
            gradients_cache.add_hook(full, enable=False)

            kl_losses = []

            for j in tqdm(
                range(0, args.nsamples, args.bsz),
                ncols=120, desc=f"Layer {i} Computing Gradients and Hessians",
                position=1, leave=False
            ):
                out = layer(inps[j: j+args.bsz].to(dev), attention_mask=batch_attention_mask, position_ids=batch_position_ids,
                            position_embeddings=batch_position_embeddings)
                logits, logits_fp = hidden2logits(out, analyzer), hidden2logits(fp_inps[j: j+args.bsz].to(dev), analyzer)

                # NLL loss
                # labels = logits_fp.argmax(dim=-1)
                labels = torch.distributions.Categorical(logits=logits_fp).sample()
                nll_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="sum",
                )
                saliency_cache.enable_hooks()
                model.zero_grad()
                nll_loss.backward(retain_graph=True)
                saliency_cache.disable_hooks()

                # top-k KL
                if args.kl_topk > 0:
                    logits_fp, indices = logits_fp.topk(args.kl_topk, dim=-1, sorted=False)
                    logits = logits.gather(-1, indices)
                kl_loss = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(logits_fp, dim=-1),
                    reduction="none",
                )
                kl_loss = kl_loss.sum(dim=-1).mean()
                gradients_cache.enable_hooks()
                model.zero_grad()
                kl_loss.backward()
                gradients_cache.disable_hooks()

                kl_losses.append(kl_loss.item())

                memory_utils.cleanup_memory()

            mean_kl_loss = sum(kl_losses) / len(kl_losses)

        saliency_cache.clear_hook()
        gradients_cache.clear_hook()

        for name in saliency_cache.names:
            saliency_cache.saliency_cache[name] = torch.cat(saliency_cache.saliency_cache[name], dim=0)
            gradients_cache.gradients_cache[name] = gradients_cache.gradients_cache[name] if i > 0 \
                                                        else gradients_cache.gradients_cache[name].zero_()

        saliency_dict = saliency_cache.saliency_cache
        gradients_dict = gradients_cache.gradients_cache

        #############################################################################

        subset = {n: full.get(n, full.get(n + ".module", None)) for n in names}

        gptq = {}
        for name in subset:
            layer_weight_bits = args.w_bits
            layer_weight_sym = not (args.w_asym)
            if "lm_head" in name:
                layer_weight_bits = 16
                continue

            gptq[name] = GPTQPlus(
                subset[name],
                saliency=saliency_dict[name], 
                gradient=gradients_dict[name],
                num_groups=args.num_groups,
                alpha=args.alpha,
                kl_loss=mean_kl_loss,
            )

            gptq[name].quantizer = quant_utils.WeightQuantizer()
            gptq[name].quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=layer_weight_sym,
                mse=args.w_clip,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            _ = layer(
                inps[j].unsqueeze(0).to(dev),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
        for h in handles:
            h.remove()

        for name in subset:
            pbar.set_postfix(module=f"layers.{i}." + name, kl=f"{mean_kl_loss:.2e}")
            layer_w_groupsize = args.w_groupsize
            gptq[name].fasterquant(
                percdamp=args.percdamp,
                groupsize=layer_w_groupsize,
                actorder=args.act_order,
                static_groups=args.act_order,
                enable_gradient_update=(i > 0),
                export_to_et=args.export_to_et,
            )
            quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
            gptq[name].free()

        for j in range(args.nsamples):
            inps[j] = layer(
                inps[j].to(dev),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0].to(inps.device)

        layers[i] = layer.to(orig_device)
        saliency_cache.clear_cache()
        gradients_cache.clear_cache()
        del layer
        del gptq
        del saliency_dict, gradients_dict
        memory_utils.cleanup_memory()

    for module in analyzer.get_pre_block_modules() + [
        analyzer.get_layernorm_before_head(),
        analyzer.get_lm_head(),
    ]:
        module.to(orig_device)
    model.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)
    logging.info("-----GPTQPlus Quantization Done-----\n")
    return quantizers
