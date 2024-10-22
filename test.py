import pandas as pd
import deepchem as dc
from lib.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from lib.models.mpnn_pom import MPNNPOMModel
import json

TASKS = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
    'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
    'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
    'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
    'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
    'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
    'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
    'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
    'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
    'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
    'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
    'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
    'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
    'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
    'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
    'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
    'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
    'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
    'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
    'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]


# | Load Train Ratio
with open('./example/train_ratios.json', 'r') as f:
    train_ratios = json.load(f)

# | Initialize Model
model = MPNNPOMModel(n_tasks=len(TASKS),
                     batch_size=128,
                     learning_rate=0.001,
                     class_imbalance_ratio=train_ratios,
                     loss_aggr_type='sum',
                     node_out_feats=100,
                     edge_hidden_feats=75,
                     edge_out_feats=100,
                     num_step_message_passing=5,
                     mpnn_residual=True,
                     message_aggregator_type='sum',
                     mode='classification',
                     number_atom_features=GraphConvConstants.ATOM_FDIM,
                     number_bond_features=GraphConvConstants.BOND_FDIM,
                     n_classes=1,
                     readout_type='set2set',
                     num_step_set2set=3,
                     num_layer_set2set=2,
                     ffn_hidden_list=[392, 392],
                     ffn_embeddings=256,
                     ffn_activation='relu',
                     ffn_dropout_p=0.12,
                     ffn_dropout_at_input_no_act=False,
                     weight_decay=1e-5,
                     self_loop=False,
                     optimizer_name='adam',
                     log_frequency=32,
                     model_dir='./trials',
                     device_name='cuda')

model.restore('./example/model.pt')

# | Inference
test_file = './example/test.csv'
df = pd.read_csv(test_file)

featurizer = GraphFeaturizer()
featurized_data = featurizer.featurize(df['SMILES'])
predictions = model.predict(dc.data.NumpyDataset(featurized_data))

df['preds'] = ''

for i, prediction in enumerate(predictions):
    high_prob_indices = [index for index, prob in enumerate(prediction) if prob > 0.65]

    selected_odors = [TASKS[index] for index in high_prob_indices]

    df.at[i, 'preds'] = ', '.join(selected_odors)

print(df)
df.to_csv('./example/test_predictions.csv', index=False)
