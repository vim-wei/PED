import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
from .imagenet_templates import IMAGENET_TEMPLATES, CUSTOM_TEMPLATES


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'PED',
                        "vision_depth": cfg.TRAINER.PED.PROMPT_DEPTH_VISION,
                        "language_depth": cfg.TRAINER.PED.PROMPT_DEPTH_TEXT, "vision_ctx": cfg.TRAINER.PED.N_CTX_VISION,
                        "language_ctx": cfg.TRAINER.PED.N_CTX_TEXT,
                        "expert": cfg.EXPERT,
                        "merge": cfg.MERGE}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'PED',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0,
                          "expert": 0,
                          "merge": "False"}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.expert = clip_model.expert

    def forward(self, prompts, tokenized_prompts, merge=False):
        
        if merge:
            
            x = prompts + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x, merge=merge)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

            return x
        else:
            all_x = []
            for i in range(self.expert):
                # import pdb
                # pdb.set_trace()
                x = prompts[i] + self.positional_embedding.type(self.dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.transformer(x, index=i)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.ln_final(x).type(self.dtype)
                # take features from the eot embedding (eot_token is the highest number in each sequence)
                x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
                all_x.append(x)
            
            return all_x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)

        # Make sure Language depth >= 1
        assert cfg.TRAINER.PED.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        n_ctx = cfg.TRAINER.PED.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PED.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # Use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :].unsqueeze(0).repeat(clip_model.expert, 1, 1)
            prompt_prefix = ctx_init
        else:
            # Random initialization
            ctx_vectors = torch.empty(clip_model.expert, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PED.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        with torch.no_grad():
            self.ZS_image_encoder = clip_model_temp.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))
        fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        self.fixed_embeddings = fixed_embeddings.cuda()

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.expert = clip_model.expert

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, merge=False):
        if merge:
            ctx = self.ctx.mean(dim=0)

            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
            prompts = self.construct_prompts(ctx, prefix, suffix)
        else:
            prompts = []
            for i in range(self.expert):
                ctx = self.ctx[i]
                if ctx.dim() == 2:
                    ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
                prefix = self.token_prefix
                suffix = self.token_suffix
                prompts.append(self.construct_prompts(ctx, prefix, suffix))

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.expert = cfg.EXPERT
        self.zeroshot_text_features = self.prompt_learner.fixed_embeddings

    def forward(self, image, label=None, index=None, merge=False):
            logit_scale = self.logit_scale.exp()
            tokenized_prompts = self.tokenized_prompts
            prompts = self.prompt_learner(merge=merge)
            text_features = self.text_encoder(prompts, tokenized_prompts, merge)
            image_features = self.image_encoder(image.type(self.dtype), index, merge)

            with autocast():
                if self.prompt_learner.training:
                    if merge:
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        logits = logit_scale * image_features @ text_features.t()
                        return logits
                    else:
                        with torch.no_grad():
                            zeroshot_image_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                            image_feat_avg = zeroshot_image_features
                            text_feat_avg = self.zeroshot_text_features
                            image_feat_avg = image_feat_avg / image_feat_avg.norm(dim=-1, keepdim=True)
                            text_feat_avg = text_feat_avg / text_feat_avg.norm(dim=-1, keepdim=True)
                        loss_ce, loss_img, loss_text = 0, 0, 0
                        logits_avg = 0
                        for i in range(self.expert):
                            img_feat = image_features[i]
                            text_feat = text_features[i]
                            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                            logits = logit_scale * img_feat @ text_feat.t()
                            loss_ce += F.cross_entropy(logits, label)
                            logits_avg += logits
                            loss_img += F.l1_loss(img_feat, image_feat_avg, reduction='mean')
                            loss_text += F.l1_loss(text_feat, text_feat_avg, reduction='mean')
                        loss_ce /= self.expert
                        loss_img /= self.expert
                        loss_text /= self.expert
                        logits_avg /= self.expert
                        loss = loss_ce + 10 * loss_img + 25 * loss_text
                        return loss, logits_avg
                else:
                    if merge:
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        logits = logit_scale * image_features @ text_features.t()
                        return logits
                    else:
                        logits = 0
                        for i in range(self.expert):
                            img_feat = image_features
                            text_feat = text_features[i]
                            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                            each_logits = logit_scale * img_feat @ text_feat.t()
                            logits += each_logits
                        logits /= self.expert
                        return logits


@TRAINER_REGISTRY.register()
class PED(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PED.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.PED.PREC == "fp32" or cfg.TRAINER.PED.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.PED.PREC == "amp" else None
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        
        self.temp = self.args.temperature
        self.collect_tesults = {"teacher": [], "student": []}

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PED.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss_pre, tea_logits = model(image, label)
            stu_logits = model(image, merge=True)
            loss_kl = F.kl_div(F.log_softmax(stu_logits / self.temp, dim=1),
                    F.softmax(tea_logits.detach() / self.temp, dim=1),
                    reduction='batchmean') * self.temp * self.temp

            loss = loss_pre + loss_kl

            optim.zero_grad()
            loss.backward()
            optim.step()
        loss_summary = {"loss": loss.item()}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)