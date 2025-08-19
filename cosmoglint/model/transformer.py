from .base import Transformer1, Transformer2, Transformer3, Transformer1WithAttn, Transformer2WithAttn, Transformer3WithAttn

def transformer_model(args, **kwargs):
    
    if "transformer" in args.model_name:
        if args.model_name == "transformer1":
            model_class = Transformer1
        elif args.model_name == "transformer2":
            model_class = Transformer2
        elif args.model_name == "transformer3":
            model_class = Transformer3
        elif args.model_name == "transformer1_with_attn":
            model_class = Transformer1WithAttn
        elif args.model_name == "transformer2_with_attn":
            model_class = Transformer2WithAttn
        elif args.model_name == "transformer3_with_attn":
            model_class = Transformer3WithAttn
        else:
            raise ValueError(f"Invalid model: {args.model_name}")
        
        model = model_class(num_condition=args.num_features_cond, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, max_length=args.max_length, num_features_in=args.num_features_in, num_features_out=args.num_features_out, **kwargs)

    else:
        raise ValueError(f"Invalid model: {args.model}")

    return model