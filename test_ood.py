import os
import sys
import copy
import pickle
from tqdm import tqdm

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sklearn.neighbors
from sklearn.metrics import RocCurveDisplay

import torch
import numpy as np

from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from catalyst.data import DistributedSamplerWrapper

import dist_utils


def test_tensor(model, device, criterion, data, target, msg=None, log_predictions=False):
    assert torch.is_tensor(data) and torch.is_tensor(target)
    
    model.eval()
    with torch.no_grad():
        output = model(data)
        loss_vals = criterion(output, target)
        test_loss = float(loss_vals.mean())
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = len(data)
    
    test_acc = 100. * correct / total
    output_dict = dict(loss=test_loss, acc=test_acc, correct=correct, total=total)
    
    loss_vals = loss_vals.detach().cpu().numpy()
    output_dict["loss_mean"] = np.mean(loss_vals)
    output_dict["loss_var"] = np.var(loss_vals)
    output_dict["loss_std"] = np.std(loss_vals)
    
    pred_dict = None
    if log_predictions:
        pred_dict = {}
        pred_dict["ex_idx"] = np.arange(len(loss_vals))
        pred_dict["loss_vals"] = loss_vals
        pred_dict["preds"] = pred.detach().cpu().numpy()
        pred_dict["targets"] = target.detach().cpu().numpy()
    
    header = "Test set" if msg is None else msg
    print(f"{header} | Loss mean: {output_dict['loss_mean']:.4f} | Loss std: {output_dict['loss_std']:.4f} | Accuracy: {test_acc:.2f}% ({correct}/{total})")
    
    return output_dict, pred_dict


def visualize_given_trajectories(trajectories_dict, output_file, label_map_dict=None, y_lim=(0., 14.), y_label="Loss values"):
    key_list = list(trajectories_dict.keys())
    num_colors = len(key_list)
    if num_colors > 9:
        cm = plt.get_cmap('hsv')
        color_list = [cm(1.*i/len(key_list)) for i in range(len(key_list))]
    elif num_colors > 4:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:pink", "tab:olive", "tab:brown", "tab:cyan"]
    else:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange"]
    assert num_colors <= len(color_list), f"{num_colors} <= {len(color_list)}"
    num_trajectories = 250
    font_size = 18

    fig, ax = plt.subplots(1, 1, figsize=(20, 8))

    handles = []
    legend_label = []

    iterator = 0
    for i, cls in enumerate(key_list):
        color = color_list[iterator]
        patch = mpatches.Patch(color=color)
        handles.append(patch)
        if label_map_dict is None:
            legend_label.append(cls.title().replace("_", " "))
        else:
            legend_label.append(label_map_dict[cls])
        
        trajectories = trajectories_dict[cls]
        assert len(trajectories) >= num_trajectories
        print(f"Class: {cls} / trajectory shape: {trajectories.shape}")

        x_axis = list(range(trajectories.shape[1]))
        all_trajs = []
        for j in range(num_trajectories):
            trajectory = trajectories[j]
            plt.plot(x_axis, trajectory, color=color_list[iterator], alpha=0.05)
            all_trajs.append(trajectory)
        
        # Plot the trajectory mean
        mean_traj = np.array(all_trajs).mean(axis=0)
        plt.plot(x_axis, mean_traj, color=color_list[iterator], alpha=0.9, linewidth=5.)
        iterator += 1
    
    ax.legend(handles, legend_label, prop={'size': font_size})
    plt.ylabel(y_label, fontsize=font_size)
    plt.xlabel("Epochs", fontsize=font_size)
    plt.ylim(y_lim[0], y_lim[1])
    plt.xlim(0., 99.)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.tight_layout()
    assert output_file is not None
    if main_proc and output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close('all')


def plot_auc(labels, predictions, key_list, output_file):
    assert isinstance(labels, dict), labels
    assert isinstance(predictions, dict), predictions
    assert isinstance(key_list, list), key_list
    
    num_colors = len(key_list)
    if num_colors > 9:
        cm = plt.get_cmap('hsv')
        color_list = [cm(1.*i/len(key_list)) for i in range(len(key_list))]
    elif num_colors > 4:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:pink", "tab:olive", "tab:brown", "tab:cyan"]
    else:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange"]
    assert num_colors <= len(color_list), f"{num_colors} <= {len(color_list)}"
    
    fontsize = 15
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    mpl.rcParams['lines.linewidth'] = 3
    plt.rcParams['axes.prop_cycle'] = cycler(alpha=[0.5])
    
    # Plot the ROC curve
    for i, k in enumerate(key_list):
        out = RocCurveDisplay.from_predictions(labels[k], predictions[k], ax=ax, name=k)
        out.line_.set_color(color_list[i])
    
    alpha = 0.8
    for l in plt.gca().lines:
        l.set_alpha(alpha)
    
    plt.xlabel("False Positive Rate", fontsize=fontsize)
    plt.ylabel("True Positive Rate", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(prop={'size': fontsize})
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", alpha=0.5)
    
    # Sort legend labels
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")


def get_loader(dataset, indices=None, batch_size=16, shuffle=False):
    sampler = None
    if indices is not None:
        sampler = torch.utils.data.SubsetRandomSampler(indices)
    if distributed:
        if sampler is not None:
            print("Using distributed sampler on top of previous sampler...")
            sampler = DistributedSamplerWrapper(sampler)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=True)
    return loader


class MaxLogitLoss(torch.nn.Module):
    """Define a custom loss criterion which only computes the max-logit trajectory"""
    def __init__(self):
        super(MaxLogitLoss, self).__init__()
    
    def forward(self, x, y):
        max_logit, max_logit_idx = x.max(dim=1)
        return max_logit


# In[ ]:


# Initialize the distributed environment
gpu = 0
world_size = 1
distributed = True  # Essential for loading pretrained models  
# distributed = int(os.getenv('WORLD_SIZE', 1)) > 1
rank = int(os.getenv('RANK', 0))
local_rank = 0

if "SLURM_NNODES" in os.environ:
    local_rank = rank % torch.cuda.device_count()
    print(f"SLURM tasks/nodes: {os.getenv('SLURM_NTASKS', 1)}/{os.getenv('SLURM_NNODES', 1)}")
elif "WORLD_SIZE" in os.environ:
    local_rank = int(os.getenv('LOCAL_RANK', 0))

if distributed:
    gpu = local_rank
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    world_size = torch.distributed.get_world_size()
    assert int(os.getenv('WORLD_SIZE', 1)) == world_size
    print(f"Initializing the environment with {world_size} processes | Current process rank: {local_rank}")

main_proc = dist_utils.is_main_proc(local_rank, shared_fs=True)
print("Is main proc?", main_proc)

dist_utils.setup_for_distributed(main_proc)


# In[ ]:


"""
Note: We can compute the loss trajectory after the completion of training only because
      we are detecting examples at test time which were not a part of training i.e. the
      model is supposed to surface examples later, similar to the minority-group exps.

OOD Dataset: Anomalous species (https://arxiv.org/pdf/1911.11132.pdf)
"""
use_max_logit_traj = True
use_model_pred = True
num_example_probes = 250


# In[ ]:


experiment_output_dir = "mapd_exp01_imagenet/"
model_file = os.path.join(experiment_output_dir, f"models_imagenet", f"model_imagenet_dynamics.pth")
data_dir = "/ds/images/imagenet/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 8
batch_size = 256
num_epochs = 100


# In[ ]:


# Create the model
num_classes = 1000
model = models.resnet50(pretrained=False, num_classes=num_classes)
model = model.to(device)
print(model)

# Convert to a distributed model
model = dist_utils.convert_to_distributed(model, local_rank, sync_bn=True)


# In[ ]:


# Initialize the test set
test_transform = [transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor()]
test_transform = transforms.Compose(test_transform)
test_set = ImageFolder(os.path.join(data_dir, "val_folders"), transform=test_transform)


# In[ ]:


print(f"Performing OOD detection using {'max-logit' if use_max_logit_traj else 'loss'} trajectory...")
if use_max_logit_traj:
    loss_fn = MaxLogitLoss()
else:
    loss_fn = torch.nn.CrossEntropyLoss()
print("OOD trajectory loss function:", loss_fn)

# Load the OOD dataset
ood_dataset_name = "imagenet_sketches"
if len(sys.argv) > 2:
    print(f"Usage: {sys.argv[0]} <Optional: dataset name>")
    exit()
if len(sys.argv) > 1:
    ood_dataset_name = sys.argv[1]
    print("Received dataset arg:", ood_dataset_name)
assert ood_dataset_name in ["anomalous_species", "imagenet_sketches", "imagenet-o"], ood_dataset_name

if ood_dataset_name == "anomalous_species":
    ood_dataset_path = "/netscratch/siddiqui/Datasets/anomalous_species/species/"
elif ood_dataset_name == "imagenet_o":
    ood_dataset_path = "/netscratch/siddiqui/Datasets/imagenet-o/"
else:
    assert ood_dataset_name == "imagenet_sketches"
    ood_dataset_path = "/netscratch/siddiqui/Datasets/imagenet-sketches/"
print(f"Dataset name: {ood_dataset_name} / Path: {ood_dataset_path}")

ood_dataset = ImageFolder(ood_dataset_path, transform=test_transform)
assert len(ood_dataset) >= num_example_probes, f"{len(ood_dataset)} < {num_example_probes}"
print(f"Number of examples found in OOD dataset ({ood_dataset_name}): {len(ood_dataset)}")

if not use_model_pred:
    # Create a class map to ensure that the class mapping is correct for ImageNet-R
    imagenet_ood_classes = ood_dataset.classes
    imagenet_classes = test_set.classes
    imagenet_cls_idx_map = {k: i for i, k in enumerate(imagenet_classes)}
    imagenet_ood_to_imagenet_cls_map = {i: imagenet_cls_idx_map[k] for i, k in enumerate(imagenet_ood_classes)}

# Create a small probe suite for the ID examples
probes_ood_det = {}
id_dataset_idx_all = np.arange(len(test_set))
id_dataset_sel_idx = np.random.choice(id_dataset_idx_all, size=num_example_probes, replace=False)
probes_ood_det.update({"id_idx": id_dataset_sel_idx})
id_dataset_test_idx = [x for x in id_dataset_idx_all if x not in id_dataset_sel_idx]
print(f"Size of original ID dataset: {len(id_dataset_idx_all)} / Probe size: {len(id_dataset_sel_idx)} / Indices left in ID dataset: {len(id_dataset_test_idx)}")

# Create a small probe suite for the OOD examples
ood_dataset_idx_all = np.arange(len(ood_dataset))
ood_dataset_sel_idx = np.random.choice(ood_dataset_idx_all, size=num_example_probes, replace=False)
probes_ood_det.update({"ood_idx": ood_dataset_sel_idx})
ood_dataset_test_idx = [x for x in ood_dataset_idx_all if x not in ood_dataset_sel_idx]
print(f"Size of original OOD dataset: {len(ood_dataset_idx_all)} / Probe size: {len(ood_dataset_sel_idx)} / Indices left in OOD dataset: {len(ood_dataset_test_idx)}")

probes_ood_det["id"] = torch.stack([test_set[i][0] for i in probes_ood_det["id_idx"]], dim=0).to(device)
if not use_model_pred:
    probes_ood_det["id_labels"] = torch.from_numpy(np.array([test_set.targets[i] for i in probes_ood_det["id_idx"]])).to(device)
    probes_ood_det["id_mem"] = torch.from_numpy(np.array([-1. for i in probes_ood_det["id_idx"]])).to(device)

probes_ood_det["ood"] = torch.stack([ood_dataset[i][0] for i in probes_ood_det["ood_idx"]], dim=0).to(device)
if not use_model_pred:
    probes_ood_det["ood_labels"] = torch.from_numpy(np.array([imagenet_ood_to_imagenet_cls_map[ood_dataset.targets[i]] for i in probes_ood_det["ood_idx"]])).to(device)
    probes_ood_det["ood_mem"] = torch.from_numpy(np.array([-1. for i in probes_ood_det["ood_idx"]])).to(device)

# Create a new test set after removing these examples from the test set
test_set_sel_loader = get_loader(test_set, id_dataset_test_idx, batch_size=batch_size)
ood_dataset_sel_loader = get_loader(ood_dataset, ood_dataset_test_idx, batch_size=batch_size)

# Load the final model
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

if use_model_pred:
    # Use the model's prediction as target for computing the trajectory
    with torch.no_grad():
        logits = model(probes_ood_det["id"])
        probes_ood_det["id_labels"] = logits.argmax(dim=1)
        
        logits = model(probes_ood_det["ood"])
        probes_ood_det["ood_labels"] = logits.argmax(dim=1)

# Create model replicas
model_replicas = {ep: copy.deepcopy(model) for ep in range(num_epochs)}

# Load all the models again one by one to compute the trajectory
id_trajectories = []
ood_trajectories = []
for epoch in range(num_epochs):
    # Specify the model path
    model_file_base, model_file_ext = os.path.splitext(model_file)
    current_model_file = f"{model_file_base}_ep_{epoch}{model_file_ext}"
    print("Loading model:", current_model_file)
    
    # Load the model
    model_replicas[epoch].load_state_dict(torch.load(current_model_file, map_location=device))
    model_replicas[epoch].eval()
    
    # Compute the loss trajectory on all probe examples
    id_stats, id_preds = test_tensor(model_replicas[epoch], device, loss_fn, probes_ood_det["id"], probes_ood_det["id_labels"],
                                     msg="In-distribution detection probe", log_predictions=True)
    ood_stats, ood_preds = test_tensor(model_replicas[epoch], device, loss_fn, probes_ood_det["ood"], probes_ood_det["ood_labels"],
                                       msg="Out-of-Distribution detection probe", log_predictions=True)
    id_trajectories.append(id_preds["loss_vals"])
    ood_trajectories.append(ood_preds["loss_vals"])

# Convert the loss values into a trajectory
id_trajectories = np.transpose(np.array(id_trajectories), (1, 0))
ood_trajectories = np.transpose(np.array(ood_trajectories), (1, 0))
print(f"ID trajectories: {id_trajectories.shape} / OOD trajectories: {ood_trajectories.shape}")

# Plot the trajectories
trajectories_dict = dict(id=id_trajectories, ood=ood_trajectories)
label_map_dict = dict(id="ID", ood="OOD")
output_file = os.path.join(experiment_output_dir, f"loss_trajectories_imagenet_id_vs_ood_{ood_dataset_name}{'_max_logit_traj' if use_max_logit_traj else ''}.png")
y_label = "Loss values"

# Compute the 90th percentile for specifying the y-axis limits
concatenated_trajs = np.concatenate([trajectories_dict[k] for k in trajectories_dict])
top_percentile = np.percentile(concatenated_trajs, 90)
print("90th percentile:", top_percentile)
y_lim = (0., top_percentile)
if use_max_logit_traj:
    y_label = "Max logit"
visualize_given_trajectories(trajectories_dict, output_file, y_lim=y_lim, y_label=y_label, label_map_dict=label_map_dict)

# Train the k-NN classifier
n_neighbors = 20
clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors)
all_labels = np.array([0 for _ in range(len(id_trajectories))] + [1 for _ in range(len(ood_trajectories))])
all_trajectories = np.concatenate([id_trajectories, ood_trajectories], axis=0)
print(f"ID trajectories: {id_trajectories.shape} / OOD trajectories: {ood_trajectories.shape} / All trajectories: {all_trajectories.shape}")
clf.fit(all_trajectories, all_labels)

# Create the one-class classifier
oc_clf_neighbors = num_example_probes
oc_clf = sklearn.neighbors.KNeighborsClassifier(oc_clf_neighbors)
id_labels = np.array([0 for _ in range(len(id_trajectories))])
oc_clf.fit(id_trajectories, id_labels)


# In[ ]:


"""
Evaluation
Compare against max-logit baseline: https://arxiv.org/pdf/1911.11132.pdf
"""
thresh = 0.5
key_list = ["ID", "OOD"]
label_dict = dict()
pred_dict = dict()
avg_dist_dict = dict()
max_logit_dict = dict()

for loader_idx, loader in enumerate([test_set_sel_loader, ood_dataset_sel_loader]):
    # Evalute the model's detection performance on the ImageNet test examples and OOD dataset
    # ID -- shouldn't discard examples; OOD -- should discard examples
    if loader_idx == 0:
        print("Evaluating on ID test set...")
    else:
        assert loader_idx == 1
        print("Evaluating on OOD test set...")
    
    predictions, targets, avg_dists, max_logits = [], [], [], []
    id_pred, ood_pred, total = 0, 0, 0
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        
        if use_model_pred:
            # Use the model's prediction as target for computing the trajectory
            model.eval()
            with torch.no_grad():
                logits = model(x)
                y = logits.argmax(dim=1)
        
        current_trajectories = []
        for epoch in range(num_epochs):
            # Compute the loss trajectory on all probe examples
            current_stats, current_preds = test_tensor(model_replicas[epoch], device, loss_fn, x, y,
                                                       msg=f"Computing loss on examples at epoch={epoch}", log_predictions=True)
            current_trajectories.append(current_preds["loss_vals"])
        
        # Convert the loss values into a trajectory
        current_trajectories = np.transpose(np.array(current_trajectories), (1, 0))
        print(f"Current trajectories: {current_trajectories.shape}")
        
        # Use the k-NN classifier to classify the given images
        traj_preds = clf.predict_proba(current_trajectories)
        assert len(traj_preds.shape) == 2, traj_preds.shape
        assert traj_preds.shape == (len(current_trajectories), 2), traj_preds.shape
        
        # Perform inference on the ID trajectories
        neighbors_dist, _ = oc_clf.kneighbors(current_trajectories)
        assert neighbors_dist.shape == (len(current_trajectories), oc_clf_neighbors), neighbors_dist.shape
        avg_dist = neighbors_dist.mean(axis=1)
        avg_dists.append(avg_dist)
        
        # Evaluate the prediction
        id_pred += np.sum(traj_preds[:, 1] < thresh)
        ood_pred += np.sum(traj_preds[:, 1] >= thresh)
        predictions.append(traj_preds[:, 1])
        total += len(current_trajectories)
        
        # Compute max-logit
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()
        with torch.no_grad():
            logits = model(x)
            max_logit, max_logit_idx = logits.max(dim=1)
            neg_max_logit = -max_logit  # Pred represents score for class 1 (OOD), but max logit is lower for OOD
            max_logits.append(neg_max_logit.cpu().detach().numpy())
    
    # Collect final stats
    predictions = np.concatenate(predictions)
    avg_dists = np.concatenate(avg_dists)
    max_logits = np.concatenate(max_logits)
    labels = np.zeros((len(predictions),), dtype=np.int64) if loader_idx == 0 else np.ones((len(predictions),), dtype=np.int64)
    
    assert id_pred == np.sum(predictions < thresh), id_pred
    assert ood_pred == np.sum(predictions >= thresh), ood_pred
    print(f">> Final stats / Dataset: {'ID' if loader_idx == 0 else 'OOD'} / Total: {total} / ID preds: {id_pred} / OOD preds: {ood_pred}")
    
    key = key_list[loader_idx]  # Can only be 0 or 1
    label_dict[key] = labels
    pred_dict[key] = predictions
    avg_dist_dict[key] = avg_dists
    max_logit_dict[key] = max_logits
    
    # Write the stats to file
    trajectory_dataset_file = os.path.join(experiment_output_dir, f"{ood_dataset_name}_{'id' if loader_idx == 0 else 'ood'}{'_max_logit' if use_max_logit_traj else ''}_trajectories.pkl")
    with open(trajectory_dataset_file, "wb") as f:
        pickle.dump([predictions, avg_dist_dict, max_logits, labels], f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Trajectory dataset written to file:", trajectory_dataset_file)

# Plot the RoC curve
key_list = ["MAP-D", "MAP-D (only ID)", "Max-Logit"]
label_dict["MAP-D"] = np.concatenate([label_dict["ID"], label_dict["OOD"]])
label_dict["MAP-D (only ID)"] = label_dict["MAP-D"].copy()
label_dict["Max-Logit"] = label_dict["MAP-D"].copy()
pred_dict["MAP-D"] = np.concatenate([pred_dict["ID"], pred_dict["OOD"]])
pred_dict["MAP-D (only ID)"] = np.concatenate([avg_dist_dict["ID"], avg_dist_dict["OOD"]])
pred_dict["Max-Logit"] = np.concatenate([max_logit_dict["ID"], max_logit_dict["OOD"]])
output_file = os.path.join(experiment_output_dir, f"auc_imagenet_id_vs_ood_{ood_dataset_name}{'_max_logit_traj' if use_max_logit_traj else ''}.png")
plot_auc(label_dict, pred_dict, key_list, output_file)

