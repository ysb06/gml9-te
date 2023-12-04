import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="train")
args = parser.parse_args()

if args.mode == "train":
    print("No trainer")
elif args.mode == "imcrts_data":
    import imc_data.imcrts
    imc_data.imcrts.IMCRTSCollector().collect()
elif args.mode == "imcrts_nodelink":
    import imc_data.imcrts
    imc_data.imcrts.IMCNodeLinkGenerator().generate()
elif args.mode == "imcrts_converting":
    import imc_data.converter
    imc_data.converter.Converter().run()
elif args.mode == "imcrts_mini":
    import imc_data.imcrts_mini
    imc_data.imcrts_mini.MiniGenerator().run()
else:
    print(f"Unknwon Mode: {args.mode}")

print("Terminating program...")
