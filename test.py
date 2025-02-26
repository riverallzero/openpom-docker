import pandas as pd
import deepchem as dc
from lib.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from lib.models.mpnn_pom import MPNNPOMModel
import json

with open("./lib/data/odor-description.txt", "r", encoding="utf-8") as file:
    TASKS = [line.strip() for line in file if line.strip()]

# | Load Train Ratio
with open("./example/train_ratios.json", "r") as f:
    train_ratios = json.load(f)

# | Initialize Model
model = MPNNPOMModel(
    n_tasks=len(TASKS),
    batch_size=128,
    learning_rate=0.001,
    class_imbalance_ratio=train_ratios,
    loss_aggr_type="sum",
    node_out_feats=100,
    edge_hidden_feats=75,
    edge_out_feats=100,
    num_step_message_passing=5,
    mpnn_residual=True,
    message_aggregator_type="sum",
    mode="classification",
    number_atom_features=GraphConvConstants.ATOM_FDIM,
    number_bond_features=GraphConvConstants.BOND_FDIM,
    n_classes=1,
    readout_type="set2set",
    num_step_set2set=3,
    num_layer_set2set=2,
    ffn_hidden_list=[392, 392],
    ffn_embeddings=256,
    ffn_activation="relu",
    ffn_dropout_p=0.12,
    ffn_dropout_at_input_no_act=False,
    weight_decay=1e-5,
    self_loop=False,
    optimizer_name="adam",
    log_frequency=32,
    model_dir="./trials",
    device_name="cuda",
)

model.restore("./example/model.pt")

# | Inference
test_file = "./example/test.csv"
df = pd.read_csv(test_file)

featurizer = GraphFeaturizer()
featurized_data = featurizer.featurize(df["SMILES"])
predictions = model.predict(dc.data.NumpyDataset(featurized_data))

df["preds"] = ""

for i, prediction in enumerate(predictions):
    high_prob_indices = [index for index, prob in enumerate(prediction) if prob > 0.65]

    selected_odors = [TASKS[index] for index in high_prob_indices]

    df.at[i, "preds"] = ", ".join(selected_odors)

print(df)
df.to_csv("./example/test_predictions.csv", index=False)
