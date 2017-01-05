require "nn"
require "optim"
require "batches"

function initialize_model_1()
    --container
    model = nn.Sequential()
    --first layer
    ----Convolution
    model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
    ----Nonlinearity
    model:add(nn.ReLU())
    ----Max-pooling
    model:add(nn.SpatialMaxPooling(2, 2))
    --Second layer
    ----Convolution
    model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
    ----Nonlinearity
    model:add(nn.ReLU())
    ----Max-Pooling
    model:add(nn.SpatialMaxPooling(2, 2))
    ----Reshape Results from the second layer
    model:add(nn.Reshape(64 * 5 *5))
    --Output layer: linear classification
    model:add(nn.Linear(64 * 5 * 5, 10))
    
    return model
end

function initialize_model_2()
    --container
    model = nn.Sequential()
    --first layer
    ----Convolution
    model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
    ----Nonlinearity
    model:add(nn.ReLU())
    ----Max-pooling
    model:add(nn.SpatialMaxPooling(2, 2))
    ----Reshape Results from the second layer
    model:add(nn.Reshape(32 * 14 * 14))
    --Output layer: linear classification
    model:add(nn.Linear(32 * 14 * 14, 10))
    
    return model
end

function initialize_model_3()
    model = nn.Sequential()
    model:add(nn.Reshape(1 * 32 * 32))
    model:add(nn.Linear(1 * 32 * 32, 10))
    return model
end

-- load data
mnistTrain = torch.load("./data/trainingData.t7") 
mnistTest  = torch.load("./data/testData.t7")
-- training parameter
training_step = 1000
train_batch_size = 100
test_batch_size  = 1000
inspection_step = 50
learning_rate = 0.05

testImages, testLabels = mnistTest:getNextBatch(test_batch_size) 
function getModelAccuracy (data, model) 
    local preds = model:forward(testImages) 
    return accuracy(preds, testLabels)
end

function trainModel (data, model, batch_size, learning_rate)
    --load image
    images, labels = data:getNextBatch(batch_size)
    --feedforward
    scores = model:forward(images)
    --define cross entropy criterion
    crit = nn.CrossEntropyCriterion()
    --use crit to calculate teh loss function
    loss = crit:forward(scores, labels)
 
    --backward
    --calculate gradient of loss w.r.t scores
    dScores = crit:backward(scores, labels) 
    --find the rest of the gradients
    model:backward(images, dScores) 
    --update parameters 
    model:updateParameters(learning_rate)
    --zero the gradients
    model:zeroGradParameters()
    --return model
    return model
end

function trainModel_L1 (data, model, batch_size, learning_rate)
    collectgarbage()
    --load image
    images, labels = data:getNextBatch(batch_size)
    --retrieve the model's parameters and gradients
    parameters,gradParameters = model:getParameters()
 
    --zero the gradients
    model:zeroGradParameters()
    
    --feedforward
    scores = model:forward(images)
    --define cross entropy criterion
    crit = nn.CrossEntropyCriterion()
    --use crit to calculate teh loss function
    loss = crit:forward(scores, labels)
    
    --backward
    --calculate gradient of loss w.r.t scores
    dScores = crit:backward(scores, labels) 
    --find the rest of the gradients
    model:backward(images, dScores) 
    --update parameters 

    local feval = function(x)
        collectgarbage()
        -- get new parameters
        if x ~= parameters then
            parameters:copy(x)
        end
        -- define L1 coef 
        coefL1 = 1e-6;
        local norm,sign= torch.norm,torch.sign
        --print('old loss'..loss)
        loss = loss + norm(coefL1 * parameters,1)
        --print('adjusted loss: '..loss)
        gradParameters:add( sign(parameters):mul(coefL1) )
        --print('L1 grad: '..norm(gradParameters,1))
        return loss, gradParameters
    end;
    
    sgdState = {
            learningRate = learning_rate,
            momentum = 0,
            learningRateDecay = 5e-7
         }
    optim.sgd(feval, parameters, sgdState)
   
    --return model
    return model
end


print('Hello, which model do you want to train?')
print('1: 2-layer CNN; 2: 1-layer CNN; 3: linear classifier')
model_id = io.read("*n")
-- Repeated Code lines, could be further cleaned 
if model_id == 1 then

    --Test Model 1
    model = initialize_model_1()
    accuracy_inspection = torch.zeros(training_step / inspection_step)
    for i = 1,training_step do 
        model = trainModel_L1(mnistTrain, model, train_batch_size, learning_rate)
        if (i % inspection_step == 0) then
            accuracy_inspection[i / inspection_step] = getModelAccuracy(mnistTest, model)
            print('['..tonumber(i / training_step * 100)..'%]')
            print(accuracy_inspection[i / inspection_step])
        end
    end
    trained_model_1 = model
    trained_model_1_acc = accuracy_inspection
    torch.save('model1.t7', trained_model_1)
    torch.save('model1_acc.t7', trained_model_1_acc)

elseif model_id == 2 then

    --Test Model 2
    model = initialize_model_2()
    accuracy_inspection = torch.zeros(training_step / inspection_step)
    for i = 1,training_step do 
        model = trainModel_L1(mnistTrain, model, train_batch_size, learning_rate)
        if (i % inspection_step == 0) then
            accuracy_inspection[i / inspection_step] = getModelAccuracy(mnistTest, model)
            print('['..tonumber(i / training_step * 100)..'%]')
            print(accuracy_inspection[i / inspection_step])
        end
    end
    trained_model_2 = model
    trained_model_2_acc = accuracy_inspection
    torch.save('model2.t7', trained_model_2)
    torch.save('model2_acc.t7', trained_model_2_acc)

elseif model_id == 3 then
    --Test Model 3
    model = initialize_model_3()
    accuracy_inspection = torch.zeros(training_step / inspection_step)
    for i = 1,training_step do 
        --model = trainModel_L1(mnistTrain, model, train_batch_size, learning_rate)
        model = trainModel(mnistTrain, model, train_batch_size, learning_rate)
        if (i % inspection_step == 0) then
            accuracy_inspection[i / inspection_step] = getModelAccuracy(mnistTest, model)
            print('['..tonumber(i / training_step * 100)..'%]')
            print(accuracy_inspection[i / inspection_step])
        end
    end
    range = torch.range(inspection_step, training_step, inspection_step)
    trained_model_3 = model
    trained_model_3_acc = accuracy_inspection
    torch.save('model3.t7', trained_model_3)
    torch.save('model3_acc.t7', trained_model_3_acc)
else 
    print("Please only choose from 1 to 3! See you next time")
end