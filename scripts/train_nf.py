from lim_mock_generator_for.nf.train import train_model, parse_args

if __name__ == "__main__":
    args = parse_args()
    train_model(args)

