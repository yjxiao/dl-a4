ok, cunn = pcall(require,'cunn')
if ok then
   LookupTable = nn.LookupTable
else
   print("Could not find cunn.")
   os.exit()
end	

require('nngraph')
require('base')
stringx = require('pl.stringx')

-- define parameters, must conform with the network setting
params = {}
params.layers = 3
params.rnn_size = 500

-- initialize model states to zeros
function setup()
    model.s = {}
    for j = 0, 1 do
        model.s[j] = {}
	for d = 1, 2 * params.layers do
	    model.s[j][d] = torch.zeros(1, params.rnn_size):cuda()
	end
    end
end

-- modified from a4_communication_loop.lua
function readline()
    local line = io.read("*line")
    if line == nil then error({code="EOF"}) end
    line = line:gsub("\n", "")
    if vocab_map[line] == nil then error({code="vocab", word = line[i]}) end
    return line
end

-- main function handling I/O
function ok_go()

    -- handshake
    io.write("OK GO\n")
    io.flush()
    
    while true do
        local ok, line = pcall(readline)
	if not ok then
	    if line.code == "EOF" then
                break -- end loop
            elseif line.code == "vocab" then
                print("Out-of-vocabulary character encountered!")
		os.exit()
	    end
	end

	-- set x and y value; error does not matter in prediction thus y can be set to any value
	local x = torch.Tensor(1):fill(vocab_map[line]):cuda()
	local y = torch.Tensor(1):fill(vocab_map[line]):cuda()
	local pred
	_, model.s[1], pred = unpack(model.core_network:forward({x, y, model.s[0]}))

	-- copy s1 to s0, which acts as prev_s input for the next iteration
	g_replace_table(model.s[0], model.s[1])

	-- write and flush prediction result
	io.write(stringx.join(" ", pred:float():storage():totable()) .. "\n")
	io.flush()
    end
end

print("Loading model ...")
model_file = "model/char_core_1h.net"
model = torch.load(model_file)
map_file = "model/char_vocab_map.tb"
vocab_map = torch.load(map_file)

setup()
ok_go()

