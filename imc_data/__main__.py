import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="train")
args = parser.parse_args()

if args.mode == "train":
    print("No trainer")
elif args.mode == "sample_generating":
    import data_loader.sample
    data_loader.sample.SampleGenerator().generate()
elif args.mode == "imcrts_data":
    import data_loader.imcrts
    data_loader.imcrts.IMCRTSCollector().collect()
elif args.mode == "imcrts_nodelink":
    import data_loader.imcrts
    data_loader.imcrts.IMCNodeLinkGenerator().generate()
else:
    print(f"Unknwon Mode: {args.mode}")

print("Terminating program...")
