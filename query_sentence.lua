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

-- define parameters, must conform with the network settings
params = {}
params.layers = 2
params.rnn_size = 200

-- initialize model states to zero
function setup()
    model.s = {}
    for j = 0, 1 do
        model.s[j] = {}
	for d = 1, 2 * params.layers do
	    model.s[j][d] = torch.zeros(1, params.rnn_size):cuda()
	end
    end
end

function readline()
    local line = io.read("*line")
    if line == nil then error({code="EOF"}) end
    line = stringx.split(line)
    if tonumber(line[1]) == nil then error({code="init"}) end
    for i = 2, #line do
        if vocab_map[line[i]] == nil then error({code="vocab", word = line[i]}) end
    end
    return line
end

-- main function handling I/O
function query_sentences()

    -- handshake
    io.write("OK GO\n")
    io.write("Query: len word1 word2 etc\n")    
    io.flush()

    while true do
        io.write("In: ")
	io.flush()
        local ok, line = pcall(readline)
	if not ok then
	    if line.code == "EOF" then
                break -- end loop
            elseif line.code == "vocab" then
                print("Out-of-vocabulary character encountered!")
		goto continue
	    elseif line.code == "init" then
	    	print("Start with a number")
		goto continue
	    end
	end

	sentence = stringx.join(" ", {select(2, unpack(line))})
	--
	local n = #line
	local l = tonumber(line[1])
	
	-- first n-1 predictions with known ground truth
	for i = 2, n-1 do
	    local x = torch.Tensor(1):fill(vocab_map[line[i]]):cuda()
	    local y = torch.Tensor(1):fill(vocab_map[line[i+1]]):cuda()
	    local pred
	    _, model.s[1], pred = unpack(model.core_network:forward({x, y, model.s[0]}))
 
            -- copy s1 to s0, which acts as prev_s input for the next iteration
	    g_replace_table(model.s[0], model.s[1])

	end
	
	-- completing the sentence
	local x = torch.Tensor(1):fill(vocab_map[line[n]]):cuda()
	local y = torch.Tensor(1):fill(vocab_map[line[n]]):cuda()
	for i = 1, l do
	    local pred
	    _, model.s[1], pred = unpack(model.core_network:forward({x, y, model.s[0]}))
	    local _, x = pred:max(2)
	    sentence = sentence .. " " .. inv_map[x[1][1]]
	    
	    g_replace_table(model.s[0], model.s[1])
	end

	-- write and flush prediction result
	io.write("Out: " .. sentence .. "\n")
	io.flush()
	
	::continue::
    end
end

print("Loading model ...")
model_file = "model/rnn_1h_word.net"
model = torch.load(model_file)
map_file = "model/word_vocab_map.tb"
vocab_map = torch.load(map_file)
inv_map_file = "model/word_inv_map.tb"
inv_map = torch.load(inv_map_file)

setup()
query_sentences()

