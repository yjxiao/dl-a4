require 'nngraph'

-- wrapper nodes for inputs
i1 = nn.Identity()()
i2 = nn.Identity()()
i3 = nn.Identity()()

-- linear node initialized with weight 1 and bias 0
linear = nn.Linear(3, 2)({i3})
linear.data.module.weight = torch.ones(2, 3)
linear.data.module.bias = torch.zeros(2)

-- rest of the structure
mul = nn.CMulTable()({i2, linear})
add = nn.CAddTable()({i1, mul})

module = nn.gModule({i1, i2, i3}, {add})

-- define inputs
x1 = torch.Tensor({1, 3})
x2 = torch.Tensor({2, 4})
x3 = torch.Tensor({1, 2, 3})

-- forward through the module
-- result should be (6, 6) x (2, 4) + (1, 3) = (13, 27)
result = module:forward({x1, x2, x3})
