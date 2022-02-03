-- function: Flatten
-- Flattens two arrays into a single array.
--
-- Parameters:
-- a - first array
-- b - second array
--
-- Returns:
-- flattened array
function Flatten(a, b)
    return a .. b
end

-- function: TableCount
-- Counts the number of elements in a table
--
-- Parameters:
-- t - the table to count
--
-- Returns:
-- the number of elements in the table
function TableCount(T)
    local count = 0
    for _ in pairs(T) do count = count + 1 end
    return count
end

-- function: CSVtoMatrix
-- takes a CSV string and returns a matrix
--
-- Parameters:
-- csv - the CSV string
--
-- Returns:
-- a matrix
--
-- Example:
-- > CSVtoMatrix("1,2,3\n4,5,6")
-- > {{1,2,3},{4,5,6}}
function CSVtoMatrix(csv)
    local rows = {}
    local row = {}
    local j = 1
    local len = string.len(csv)

    for i = 1, len do
        local c = csv:sub(i, i)
        if c == ',' then
            j = j + 1
            row[j] = ""
        elseif c == '\n' then
            table.insert(rows, row)
            row = {}
            j = 1
        elseif c ~= ' ' and c ~='\r' then
            row[j] = c
        end
    end
    table.insert(rows, row)

    return rows
end

-- function: LoadFile
-- loads a file
--
-- Parameters:
-- path - the path of the file
--
-- Returns:
-- the contents of the file
function LoadFile(path)
    local f = io.open(path, "r")
    local content = f:read("*all")
    f:close()
    return content
end

-- function: PrintArray
-- prints an array or matrix
--
-- Parameters:
-- arr - the array or matrix
-- level - always nil (for recursive calls)
--
-- Returns:
-- nothing
function PrintArray(arr, level)
    local str = ""

    if level == nil then
        print(PrintArray(arr, 0))
        return
    end

    for index,value in pairs(arr) do
        if type(value) == "table" then
            str = str .. PrintArray(value, (level + 1))
        else 
            str = str .. value .. ", "
        end
    end
    return "\n{" .. str .. "}"
end

-- function: Divisable
-- returns true if the number is divisable by the divisor
--
-- Parameters:
-- a - the first number
-- b - the second number
--
-- Returns:
-- true if the number is divisable by the divisor
function Divisable(a, b)
    return math.floor(a / b) == a / b
end

-- function: PrintArrayAsImageToConsole
-- prints an array or matrix as an image to the console
--
-- Parameters:
-- arr - the array
-- width - the width of the image
--
-- Returns:
-- nothing
function PrintArrayAs2DImageToConsole(arr, width)
    OneChar = "██"
    ZeroChar = "  "
    local str = ""
    for i = 1, TableCount(arr) do
        if arr[i] == "1" or arr[i] == 1 then
            str = str .. OneChar
        else
            str = str .. ZeroChar
        end

        if Divisable(i, width) then
            str = str .. "\n"
        end
    end
    print(str)
end


-- function: RandomWeights
-- returns a matrix of random weights
--
-- Parameters:
-- rows - the number of rows
-- columns - the number of columns
--
-- Returns:
-- a matrix of random weights
function RandomWeights(rows, columns)
    local weights = {}
    for i = 1, rows do
        weights[i] = {}
        for j = 1, columns do
            weights[i][j] = math.random()
        end
    end
    return weights
end

-- function: sigmoid
-- returns the sigmoid of a number
--
-- Parameters:
-- x - the number
--
-- Returns:
-- the sigmoid of the number
function Sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

-- function: ArraySigmoid
-- returns the sigmoid of an array
--
-- Parameters:
-- arr - the array
--
-- Returns:
-- the sigmoid of the array
function ArraySigmoid(arr)
    local newArr = {}
    for i = 1, TableCount(arr) do
        newArr[i] = Sigmoid(arr[i])
    end
    return newArr
end

-- function: CalculateLoss
-- calculates the loss of a neural network by mean squared error
--
-- Parameters:
-- y - the expected output
-- yHat - the predicted output
--
-- Returns:
-- the loss of the neural network
function CalculateLoss(y, yHat)
    local loss = 0
    for i = 1, TableCount(y) do
        loss = loss + (y[i] - yHat[i]) ^ 2
    end
    return loss / TableCount(y)
end

-- function: ArraySquareCost
-- calculates the square cost of an array
--
-- Parameters:
-- y - the expected output
-- yHat - the predicted output
--
-- Returns:
-- the square cost of the array
function ArraySquareCost(y, yHat)
    local cost = {}
    for i = 1, TableCount(y) do
        cost[i] = (y[i] - yHat[i]) ^ 2
    end
end

-- function: ArraySumProduct
-- calculates the sum product of two arrays
-- Matrix multiplication of one dimensional arrays
--
-- Parameters:
-- a - the first array
-- b - the second array
--
-- Returns:
-- the sum product of the two arrays
function ArraySumProduct(a, b)
    local dot = {}
    for i = 1, TableCount(a) do
        for j = 1, TableCount(b) do
            dot[i] = dot[i] + a[i][j] * b[i][j]
        end
    end
    return dot
end

-- function: FForward
-- Creates a forward propagation of a neural network
--
-- Parameters:
-- x - the input layer
-- w1 - the weights of the first layer
-- w2 - the weights of the second layer
-- o - the output layer
--
-- Returns:
-- the output of the neural network
function FForward(x, w1, w2, o)
    -- input layer x

    -- hidden layer w1 * x
    local z1 = ArraySumProduct(x, w1)
    local a1 = ArraySigmoid(z1)

    -- hidden layer 2 w2 * a1
    local z2 = ArraySumProduct(a1, w2)
    local a2 = ArraySigmoid(z2)

    -- output layer o * a2
    local z3 = ArraySumProduct(a2, o)
    local a3 = ArraySigmoid(z3)
    return a3
end

-- function: BackProp
-- Creates a back propagation of a neural network
--
-- Parameters:
-- x - the input layer
-- y - the expected output
-- w1 - the weights of the first layer
-- w2 - the weights of the second layer
-- o - the output layer
--
-- Returns:
-- new weights w1, w2 and w3 of the neural network
function BackProp(x, y, w1, w2, o, alpha)

    -- hidden layer
    local z1 = ArraySumProduct(x, w1)
    local a1 = ArraySigmoid(z1)

    -- hidden layer 2
    local z2 = ArraySumProduct(a1, w2)
    local a2 = ArraySigmoid(z2)

    -- output layer
    local z3 = ArraySumProduct(a2, o)
    local a3 = ArraySigmoid(z3)

    -- output layer error
    local o_error = {}
    for i = 1, TableCount(y) do
        o_error[i] = y[i] - a3[i]
    end

    local o_cost = ArraySquareCost(y, a3)



    return {w1, w2, o}
end


print("MLToolset loaded")

Data = LoadFile("data.csv")
Matrix = CSVtoMatrix(Data)
Labels = {
{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
}
GroundTruth = { "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" }
-- for i = 1, TableCount(Matrix) do
--     PrintArrayAs2DImageToConsole(Matrix[i], 6)
-- end