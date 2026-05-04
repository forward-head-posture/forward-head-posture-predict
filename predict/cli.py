from predict.app import load_model, main


def cli():
    model = load_model()
    main(model)
