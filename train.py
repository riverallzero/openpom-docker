import deepchem as dc
from lib.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from lib.utils.data_utils import get_class_imbalance_ratio
from lib.models.mpnn_pom import MPNNPOMModel
from datetime import datetime
import json
import os

output_dir = "./trials"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open("./lib/data/odor-description.txt", "r", encoding="utf-8") as file:
    TASKS = [line.strip() for line in file if line.strip()]

# | Get Dataset
input_file = "./lib/data/GSLF.csv"

featurizer = GraphFeaturizer()
smiles_field = "nonStereoSMILES"
loader = dc.data.CSVLoader(
    tasks=TASKS, feature_field=smiles_field, featurizer=featurizer
)
dataset = loader.create_dataset(inputs=[input_file])
n_tasks = len(dataset.tasks)

# | GET Train/Valid/Test Splits
randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
train_dataset, test_dataset, valid_dataset = (
    randomstratifiedsplitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=1
    )
)

train_ratios = get_class_imbalance_ratio(train_dataset)
assert len(train_ratios) == n_tasks

with open(os.path.join(output_dir, "train_ratios.json"), "w") as f:
    json.dump(train_ratios, f)

learning_rate = 0.001
nb_epoch = 20
metrics_name = "roc_auc_score"
n_classes = 1

print("\n")
print("TRAIN Info ===============================")
print(f'  o Data: {input_file.split("/")[-1]}')
print(f"  o No of tasks: {len(TASKS)}")
print(f"  o No of dataset: {len(dataset)}")
print("  o Data Split")
print(f"    o train: {len(train_dataset)}")
print(f"    o valid: {len(valid_dataset)}")
print(f"    o test: {len(test_dataset)}")
print("  o Parameter")
print(f"    o lr = {learning_rate}")
print(f"    o epoch = {nb_epoch}")
print(f"    o metrics = {metrics_name}")
print("==========================================")
print("\n")

# | Initialize Model
model = MPNNPOMModel(
    n_tasks=n_tasks,
    batch_size=128,
    learning_rate=learning_rate,
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
    n_classes=n_classes,
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

metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

start_time = datetime.now()
for epoch in range(1, nb_epoch + 1):
    loss = model.fit(
        train_dataset,
        nb_epoch=1,
        max_checkpoints_to_keep=1,
        deterministic=False,
        restore=epoch > 1,
    )
    train_scores = model.evaluate(train_dataset, [metric])[metrics_name]
    valid_scores = model.evaluate(valid_dataset, [metric])[metrics_name]
    print(
        f"epoch {epoch}/{nb_epoch} ; loss = {loss}; train = {train_scores}; val = {valid_scores}"
    )
model.save_checkpoint()
end_time = datetime.now()

test_scores = model.evaluate(test_dataset, [metric])[metrics_name]

print("\n")
print("TEST Info =================================")
print(f"  o time taken: {str(end_time - start_time)}")
print(f"  o {metrics_name} = {test_scores}")
print("===========================================")
