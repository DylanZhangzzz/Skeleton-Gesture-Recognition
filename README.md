# Attention-based skeleton-gesture-recognition with 2D position encoding
![hand_model](https://github.com/DylanZhangzzz/Skeleton-Gesture-Recognication/assets/42655081/7adbaa36-8381-4fd2-a3b8-6ecfb1c8020f)

The traditional methods usually flatten 2D or 3D hand joint data into a 1D tensor for training, it is difficult for the model to learn the physical connections between joints. 
Considering the complexity of the skeletal structure of the hand, it is naturally embedded in the form of an ordered structure rather than a vector sequence. The above figure demonstrates the hand skeleton model and corresponding joint table under joint and finger hierarchy. 
The purpose of 2D position encoding is to convey the structure of a skeleton in a 2D space, where $(x,y)$ represents a point. The encoding method involves using the same encoding for channel features from the same joints across multiple frames. This helps to provide additional spatial information about channel features from the same finger (column) or the same joint levels (row). The integer values of i and j range from 0 to $D/4$, where $D$ is the size of the channel dimension.
![att_network](https://github.com/DylanZhangzzz/Skeleton-Gesture-Recognication/assets/42655081/216ad513-a6f5-4c87-9d59-6d3266c70139)

The spatial and temporal attention modules are constructed on the same attention module but use various feature dimension sizes. The input feature size had illustrated in the net_overview fig, which $X_{in} \in R^{T \times NC}$ and $X_{in} \in R^{N \times TC}$ for the temporal and spatial module, respectively, where $T$ is the length of the frame, $N$ numbers of skeleton nodes, $C$ channel of each node.
