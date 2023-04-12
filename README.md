# Metadata Archaeology [ICLR'23 Spotlight]

A PyTorch implementation for the ***ICLR'23*** paper [***Metadata Archaeology: Unearthing Data Subsets by Leveraging Training Dynamics***](https://arxiv.org/abs/2209.10015).

Check the [***OpenReview***](https://openreview.net/forum?id=PvLnIaJbt9) page for detailed discussions on the paper.

## Execution

The code can be executed using the ```train.sh``` script provided.

The code was tested with NVIDIA PyTorch container 22.10 (https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-10.html).

The code also requires memorization/consistency scores to compute typical/atypical examples.
Configure the ```data_dir``` variable in the script to define the path to these scores.

We use consistency scores by default. Consistency scores can be downloaded from the official website: ```https://pluskid.github.io/structural-regularity/```

Similarly, memorization scores can also be downloaded from the official website: ```https://pluskid.github.io/influence-memorization/```

## Surfaced Examples

Surfaced examples can be accessed through [***Google Drive***](https://drive.google.com/drive/folders/1TXGEHBNxcRUFm-qR-I2241IOgnAmqcVR).
Also checkout our [***webpage***](https://metadata-archaeology.github.io/) for more details.

## Loss Trajectories

As loss trajectories themselves as of great interest, the computed loss trajectories on CIFAR-10, CIFAR-100 and ImageNet are directly available for download through 
[***Google Drive***](https://drive.google.com/drive/folders/1Hds32eyIuGSJd1e6ZndC6OEcwfFWz4vK?usp=share_link).
Please refer to the main script ```metadata_archaeology.py``` for details regarding how to load these files.

## Citation

```
@article{siddiqui2022metadataarchaeology,
  title={Metadata Archaeology: Unearthing Data Subsets by Leveraging Training Dynamics},
  author={Siddiqui, Shoaib Ahmed and Rajkumar, Nitarshan and Maharaj, Tegan and Krueger, David and Hooker, Sara},
  journal={arXiv preprint},
  year={2022},
  url={https://arxiv.org/abs/2209.10015}
}
```

## Disclaimer

This branch contains minor deviations from the initial code.
Please check the `initial` branch when attempting to reproduce results.

## Issues/Feedback

In case of any issues, feel free to drop me an email or open an issue on the repository.

Email: **msas3@cam.ac.uk**

## License

MIT
