# Metadata Archaeology

A PyTorch implementation for the paper [***Metadata Archaeology: Unearthing Data Subsets by Leveraging Training Dynamics***](https://arxiv.org/abs/2209.10015).

## Execution

The code can be executed using the ```train.sh``` script provided.
The code also requires memorization/consistency scores to compute typical/atypical examples.
Configure the ```data_dir``` variable in the script to define the path to these scores.

We use consistency scores by default. Consistency scores can be downloaded from the official website: ```https://pluskid.github.io/structural-regularity/```

Similarly, memorization scores can also be downloaded from the official website: ```https://pluskid.github.io/influence-memorization/```

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

## Issues/Feedback:

In case of any issues, feel free to drop me an email or open an issue on the repository.

Email: **msas3@cam.ac.uk**

## License:

MIT
