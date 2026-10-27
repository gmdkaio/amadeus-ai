import argparse


def main():
    parser = argparse.ArgumentParser(prog="amadeus", description="Talk to Kurisu.")
    parser.add_argument("--no-voice", action="store_true", help="text only, skip GPT-SoVITS")
    args = parser.parse_args()

    # Imported here so --help doesn't wait on torch/unsloth
    from amadeus import chat
    chat.run(voice=not args.no_voice)


if __name__ == "__main__":
    main()
