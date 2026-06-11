from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
NB = ROOT / "Scam_or_Not_Scam_IR_Project.ipynb"


def cell(kind, source):
    data = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        data.update({"execution_count": None, "outputs": []})
    return data


nb = json.loads(NB.read_text(encoding="utf-8"))
existing = "".join("".join(c.get("source", [])) for c in nb["cells"])

if "Real-World Dataset Collection" not in existing:
    nb["cells"].extend(
        [
            cell(
                "markdown",
                "## Real-World Dataset Collection\n\nTo make the project more applicable, we add a new dataset layer beyond the original clean assignment corpus. The new layer combines public SMS spam data, public-warning scam patterns, Khmer manual examples, synthetic realistic scam templates, and safe educational contrast messages.",
            ),
            cell(
                "code",
                "from pathlib import Path\nimport json\nimport pandas as pd\nreal_world_path = Path('data/real_world_message_dataset.csv')\nsource_meta_path = Path('data/real_world_dataset_sources.json')\nreal_world_df = pd.read_csv(real_world_path)\nsource_meta = json.loads(source_meta_path.read_text(encoding='utf-8'))\nprint('Rows:', len(real_world_df))\ndisplay(real_world_df['label'].value_counts().to_frame('count'))\ndisplay(real_world_df['source'].value_counts().to_frame('count'))\ndisplay(real_world_df.sample(10, random_state=42)[['label', 'scam_type', 'language', 'source', 'text']])",
            ),
            cell(
                "markdown",
                "### Dataset Ethics\n\nOnly public datasets and manually/synthetically authored non-private messages are used. Private SMS, personal chat histories, login-protected pages, and identifiable user data should not be scraped. Real user feedback should be collected only with consent and should remove sensitive information before retraining.",
            ),
        ]
    )

NB.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Added dataset collection cells.")
