import json

from pathlib import Path
from tqdm import tqdm


def main():
    """
    Utility function to split the data into GS6 and GS6_guidance files
    :return:
    """
    for split in ["train", "valid", "test"]:
        file_path = Path("data/Fully_Augmented_Dataset/{}.json".format(split))
        out_file_path = Path("data/Fully_Augmented_Dataset/GS6/{}.json".format(split))
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        out_guidance_file_path = Path("data/Fully_Augmented_Dataset/GS6_guid/{}.json".format(split))
        out_guidance_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "r") as f_in, open(out_file_path, "w") as f_out, open(out_guidance_file_path, "w") as f_out_guid:
            for line in tqdm(f_in, desc="Processing {}".format(split)):
                data = json.loads(line)
                guidance = data.pop("guidance", None)
                if guidance:
                    guidance_doc = {
                        "id": data["id"],
                        "document": guidance
                    }
                    f_out_guid.write(json.dumps(guidance_doc) + "\n")
                else:
                    raise ValueError("Guidance not found in data")
                f_out.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
