
class ConformPixartX():

    def _perform_iterative_refinement_step(
        self,
        # latents: torch.Tensor,
        # token_groups: List[List[int]],
        # loss: torch.Tensor,
        text_embeddings: torch.Tensor,
        # step_size: float,
        t: int,
        # refinement_steps: int = 20,
        # do_smoothing: bool = True,
        # smoothing_kernel_size: int = 3,
        # smoothing_sigma: float = 0.5,
        # temperature: float = 0.07,
        # softmax_normalize: bool = True,
        # softmax_normalize_attention_maps: bool = False,
        attention_maps_t_plus_one: Optional[torch.Tensor] = None,
        # loss_fn: str = "ntxent",
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
        according to our loss objective until the given threshold is reached for all tokens.
        """
        for iteration in range(self.config.refinement_steps):
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            # attention_maps = self._aggregate_attention()
            
            self.attn_fetch_x.store_attn_by_timestep(t,self.unet)
            attention_maps = self.attn_fetch_x.storage

            loss = self._compute_contrastive_loss(
                attention_maps=attention_maps,
                attention_maps_t_plus_one=attention_maps_t_plus_one,
                token_groups=token_groups,
                loss_type=loss_fn,
                do_smoothing=do_smoothing,
                temperature=temperature,
                smoothing_kernel_size=smoothing_kernel_size,
                smoothing_sigma=smoothing_sigma,
                softmax_normalize=softmax_normalize,
                softmax_normalize_attention_maps=softmax_normalize_attention_maps,
            )

            if loss != 0:
                # print("update refine")
                latents = self._update_latent(latents, loss, step_size)

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        _ = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        self.attn_fetch_x.store_attn_by_timestep(t,self.unet)
        attention_maps = self.attn_fetch_x.storage

        loss = self._compute_contrastive_loss(
            attention_maps=attention_maps,
            attention_maps_t_plus_one=attention_maps_t_plus_one,
            token_groups=token_groups,
            loss_type=loss_fn,
            do_smoothing=do_smoothing,
            temperature=temperature,
            smoothing_kernel_size=smoothing_kernel_size,
            smoothing_sigma=smoothing_sigma,
            softmax_normalize=softmax_normalize,
            softmax_normalize_attention_maps=softmax_normalize_attention_maps,
        )
        return loss, latents
        @staticmethod
    def _compute_contrastive_loss(
        attention_maps: torch.Tensor,
        attention_maps_t_plus_one: Optional[torch.Tensor],
        token_groups: List[List[int]],
        loss_type: str,
        temperature: float = 0.07,
        do_smoothing: bool = True,
        smoothing_kernel_size: int = 3,
        smoothing_sigma: float = 0.5,
        softmax_normalize: bool = True,
        softmax_normalize_attention_maps: bool = False,
    ) -> torch.Tensor:
        """Computes the attend-and-contrast loss using the maximum attention value for each token."""

        attention_for_text = attention_maps[:, :, 1:-1]

        if softmax_normalize:
            attention_for_text *= 100
            attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        attention_for_text_t_plus_one = None
        if attention_maps_t_plus_one is not None:
            attention_for_text_t_plus_one = attention_maps_t_plus_one[:, :, 1:-1]
            if softmax_normalize:
                attention_for_text_t_plus_one *= 100
                attention_for_text_t_plus_one = torch.nn.functional.softmax(
                    attention_for_text_t_plus_one, dim=-1
                )

        indices_to_clases = {}
        for c, group in enumerate(token_groups):
            for obj in group:
                indices_to_clases[obj] = c

        classes = []
        embeddings = []
        for ind, c in indices_to_clases.items():
            classes.append(c)
            # Shift indices since we removed the first token
            embedding = attention_for_text[:, :, ind - 1]
            if do_smoothing:
                smoothing = GaussianSmoothing(
                    kernel_size=smoothing_kernel_size, sigma=smoothing_sigma
                ).to(attention_for_text.device)
                input = F.pad(
                    embedding.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                )
                embedding = smoothing(input).squeeze(0).squeeze(0)
            embedding = embedding.view(-1)

            if softmax_normalize_attention_maps:
                embedding *= 100
                embedding = torch.nn.functional.softmax(embedding)
            embeddings.append(embedding)

            if attention_for_text_t_plus_one is not None:
                classes.append(c)
                # Shift indices since we removed the first token
                embedding = attention_for_text_t_plus_one[:, :, ind - 1]
                if do_smoothing:
                    smoothing = GaussianSmoothing(
                        kernel_size=smoothing_kernel_size, sigma=smoothing_sigma
                    ).to(attention_for_text.device)
                    input = F.pad(
                        embedding.unsqueeze(0).unsqueeze(0),
                        (1, 1, 1, 1),
                        mode="reflect",
                    )
                    embedding = smoothing(input).squeeze(0).squeeze(0)
                embedding = embedding.view(-1)

                if softmax_normalize_attention_maps:
                    embedding *= 100
                    embedding = torch.nn.functional.softmax(embedding)
                embeddings.append(embedding)

        classes = torch.tensor(classes).to(attention_for_text.device)
        embeddings = torch.stack(embeddings, dim=0).to(attention_for_text.device)

        # loss_fn = losses.NTXentLoss(temperature=temperature)

        if loss_type == "ntxent_contrastive":
            if len(token_groups) > 0 and len(token_groups[0]) > 1:
                loss_fn = losses.NTXentLoss(temperature=temperature)
            else:
                loss_fn = losses.ContrastiveLoss(
                    distance=distances.CosineSimilarity(), pos_margin=1, neg_margin=0
                )
        elif loss_type == "ntxent":
            loss_fn = losses.NTXentLoss(temperature=temperature)
        elif loss_type == "contrastive":
            loss_fn = losses.ContrastiveLoss(
                distance=distances.CosineSimilarity(), pos_margin=1, neg_margin=0
            )
        else:
            raise ValueError(f"loss_fn {loss_type} not supported")

        loss = loss_fn(embeddings, classes)
        return loss
    def do_conform(self,):

        loss = self._compute_contrastive_loss(
            attention_maps=attn_map,
            attention_maps_t_plus_one=attention_map_t_plus_one,
            token_groups=token_group,
            loss_type=loss_fn,
            temperature=temperature,
            do_smoothing=do_smoothing,
            smoothing_kernel_size=smoothing_kernel_size,
            smoothing_sigma=smoothing_sigma,
            softmax_normalize=softmax_normalize,
            softmax_normalize_attention_maps=softmax_normalize_attention_maps,
        )


    # print("attn loss", loss)

    # If this is an iterative refinement step, verify we have reached the desired threshold for all
        if i in iterative_refinement_steps:

            loss, latent = self._perform_iterative_refinement_step(
                latents=latent,
                token_groups=token_group,
                loss=loss,
                text_embeddings=text_embedding,
                step_size=step_size[i],
                t=t,
                refinement_steps=refinement_steps,
                do_smoothing=do_smoothing,
                smoothing_kernel_size=smoothing_kernel_size,
                smoothing_sigma=smoothing_sigma,
                temperature=temperature,
                softmax_normalize=softmax_normalize,
                softmax_normalize_attention_maps=softmax_normalize_attention_maps,
                attention_maps_t_plus_one=attention_map_t_plus_one,
                loss_fn=loss_fn,
                )


    # Perform gradient update
        if i < max_iter_to_alter:
            if loss != 0:
                print("update latent")
                latent = self._update_latent(
                    latents=latent,
                    loss=loss,
                    step_size=step_size[i],
                )