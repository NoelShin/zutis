if __name__ == '__main__':
    import os
    from argparse import ArgumentParser, Namespace
    from typing import Optional
    import json
    import yaml
    import torch
    from criterion import Criterion
    from utils import (
        get_dataset, get_experim_name, get_network, get_optimiser, get_lr_scheduler, get_palette,  set_seed,
        get_label_id_to_category
    )
    from utils.running_score import RunningScore
    from utils.visualiser import Visualiser
    from trainer import Trainer

    # parse arguments
    parser = ArgumentParser("ZUTIS")
    parser.add_argument("--p_config", type=str, default="", required=True)
    parser.add_argument("--p_state_dict", type=str, default='')
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--seed", "-s", default=0, type=int)
    parser.add_argument("--weight_ce_loss", type=float, default=1.0)
    parser.add_argument("--suffix", type=str, default='')
    args = parser.parse_args()

    args: Namespace = parser.parse_args()
    base_args = yaml.safe_load(open(f"{args.p_config}", 'r'))

    args: dict = vars(args)
    args.update(base_args)
    args: Namespace = Namespace(**args)
    set_seed(args.seed)

    experim_name: str = get_experim_name(args)

    if args.dataset_name == "imagenet-s":
        dir_ckpt: str = f"{args.dir_ckpt}/{args.dataset_name}{args.n_categories}/{args.split}/{experim_name}"
    else:
        dir_ckpt: str = f"{args.dir_ckpt}/{args.dataset_name}/{args.split}/{experim_name}"
    dir_dt_masks = f"{dir_ckpt}/dt"

    if os.path.exists(f"{dir_dt_masks}/final_model.pt") and args.p_state_dict is None:
        print(f"already final model exists at {dir_dt_masks}/final_model.pt.")
        exit(0)

    os.makedirs(dir_dt_masks, exist_ok=True)

    print(f"\n====={dir_ckpt} is created.=====\n")
    json.dump(vars(args), open(f"{dir_ckpt}/config.json", 'w'), indent=2, sort_keys=True)

    # device setting
    device: torch.device = torch.device("cuda:0")

    # instantiate a validation dataloader
    val_dataloader = get_dataset(
        dir_dataset=args.dir_val_dataset,
        dataset_name=args.dataset_name,
        split=args.split,
        categories=args.categories,
        n_categories=args.n_categories,
        **args.val_dataloader_kwargs
    )

    try:
        encoder_type: Optional[str] = args.encoder_type  # default: "clip"
    except AttributeError:
        encoder_type = None

    try:
        frozen_bn: Optional[bool] = args.frozen_bn  # default: True
    except AttributeError:
        frozen_bn = None

    try:
        stop_gradient: Optional[bool] = args.stop_gradient  # default: True
    except AttributeError:
        stop_gradient = None

    try:
        decoder_image_n_dims: Optional[int] = args.decoder_image_n_dims  # default: True
    except AttributeError:
        decoder_image_n_dims = None

    # instantiate a segmentation network
    network = get_network(
        network_name=args.clip_arch,
        encoder_type=encoder_type,
        categories=args.categories,
        frozen_bn=frozen_bn,
        stop_gradient=stop_gradient,
        decoder_image_n_dims=decoder_image_n_dims
    ).to(device)

    # instantiate a visualiser
    palette = get_palette(dataset_name=args.dataset_name, n_categories=args.n_categories)
    visualiser = Visualiser(
        label_id_to_category=get_label_id_to_category(
            dataset_name=args.dataset_name,
            n_categories=args.n_categories if args.dataset_name == "imagenet-s" else None
        )
    )

    # instantiate a trainer
    trainer = Trainer(
        network=network, device=device, dir_ckpt=dir_dt_masks, palette=palette, visualiser=visualiser, debug=args.debug
    )

    if args.p_state_dict == '':
        try:
            random_duplicate: bool = args.random_duplicate  # default: True
        except AttributeError:
            random_duplicate = False

        # instantiate a training dataloader
        train_dataloader = get_dataset(
            dataset_name=args.index_dataset_name,
            dir_dataset=args.dir_train_dataset,
            split="train",  # for index dataset == "imagenet",
            p_filename_to_image_embedding=args.p_filename_to_image_embedding,  # for index dataset == "index"
            image_size=args.train_image_size,
            ignore_index=args.ignore_index,
            categories=args.categories,
            category_to_p_images_fp=args.category_to_p_images_fp,
            n_images=args.n_images,
            scale_range=args.scale_range,
            use_advanced_copy_paste=args.use_advanced_copy_paste,
            n_categories=args.n_categories if args.dataset_name == "imagenet-s" else None,
            random_duplicate=random_duplicate,
            **args.train_dataloader_kwargs
        )

        # instantiate a loss function
        criterion = Criterion(
            text_embeddings=network.text_embeddings,
            ignore_index=args.ignore_index,
            weight_ce_loss=args.weight_ce_loss
        )

        # instantiate a metric meter
        metric_meter = RunningScore(val_dataloader.dataset.n_categories)

        # instantiate an optimiser
        optimiser = get_optimiser(network=network)

        # instantiate a learning rate scheduler
        lr_scheduler = get_lr_scheduler(optimiser=optimiser, n_iters=args.n_iters)

        trainer.fit(
            dataloader=train_dataloader,
            criterion=criterion,
            optimiser=optimiser,
            n_iters=args.n_iters,
            lr_scheduler=lr_scheduler,
            metric_meter=metric_meter,
            iter_eval=args.iter_eval,
            iter_log=args.iter_log,
            val_dataloader=val_dataloader
        )
    else:
        trainer.evaluate(dataloader=val_dataloader, p_state_dict=args.p_state_dict)
