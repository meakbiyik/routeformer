# Using `GEMDataset` Class

The `GEMDataset` class is a subclass of the `torch.utils.data.IterableDataset` class. It is used to load the Routeformer dataset. An example of how to use it is shown below.

```python
from routeformer import GEMDataset

dataset = GEMDataset(
    "/data/routeformer",
    input_length=3,
    target_length=1,
    step_size=1,
    num_workers=8
)
model = BasicModel()

# define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# train the network
print("Starting Training")
running_loss = 0.0
for i, data in enumerate(dataset):
    # get the inputs
    left = data["train"]["left_video"].unsqueeze(0).permute(0, 2, 1, 3, 4)
    right = data["train"]["right_video"].unsqueeze(0).permute(0, 2, 1, 3, 4)
    gps = data["train"]["gps"][:, :2]
    target_gps = data["target"]["gps"][:, :2]
    optimizer.zero_grad()
    outputs = model(left, right, gps)
    loss = criterion(outputs, target_gps)
    loss.backward()
    optimizer.step()
```

See the definition of the `GEMDataset` class in `routeformer/io/dataset.py` for more details and options. For the full example, see `examples/dataset_demo.py`.
