require "nn"
require 'gnuplot'
require 'util'

training_step = 1000
inspection_step = 50

range = torch.range(inspection_step, training_step, inspection_step)

--load models
model_1 = torch.load('model1.t7')
model_1_acc = torch.load('model1_acc.t7')

model_2 = torch.load('model2.t7')
model_2_acc = torch.load('model2_acc.t7')

model_3 = torch.load('model3.t7')
model_3_acc = torch.load('model3_acc.t7')

model_4 = torch.load('model3_L1.t7')
model_4_acc = torch.load('model3_L1_acc.t7')

--plot learning curve
gnuplot.figure(1)
gnuplot.plot({'TwoLayer',range, model_1_acc,'+-'},
    {'OneLayer',range, model_2_acc,'+-'},
    {'Linear',range, model_3_acc,'+-'},
    {'Linear L1-regularization',range, model_4_acc,'+-'})
gnuplot.movelegend('left','top')

--load image
image_center = torch.load('./data/translations/center.t7')
image_left   = torch.load('./data/translations/leftShifts.t7')
image_right  = torch.load('./data/translations/rightShifts.t7')

function getScore(model, image)
    model:forward(image)
    scores = model.output:clone()
    return scores
end
function getDiff(model)
    score_center = getScore(model,image_center)
    score_all  = {}
    diff       = {}
    diff_image = {}
    invariance = torch.zeros(11)
    for i = 1,5 do
        score_all[6 - i] = getScore(model,image_left[i])
    end
    score_all[6] = score_center
    for i = 1,5 do
        score_all[i + 6] = getScore(model,image_right[i])
    end
    for i = 1,11 do
        invariance[i] = (score_all[i] - score_center):norm(1, 2):mean()
    end;
    return invariance
end
function getDiff_L2(model)
    score_center = getScore(model,image_center)
    score_all  = {}
    diff       = {}
    diff_image = {}
    invariance = torch.zeros(11)
    for i = 1,5 do
        score_all[6 - i] = getScore(model,image_left[i])
    end
    score_all[6] = score_center
    for i = 1,5 do
        score_all[i + 6] = getScore(model,image_right[i])
    end
    for i = 1,11 do
        invariance[i] = avgDistance(score_all[i], score_center)
    end
    return invariance
    --return score_all, score_center
end

--plot unnormalized invariance
invariance_1 = getDiff_L2(model_1)
invariance_2 = getDiff_L2(model_2)
invariance_3 = getDiff_L2(model_3)
invariance_4 = getDiff_L2(model_4)
gnuplot.figure(2)
gnuplot.plot({'TwoLayer',invariance_1,'+-'},
    {'OneLayer', invariance_2,'+-'},
    {'Linear', invariance_3,'+-'},
    {'Linear L1-regularization', invariance_4,'+-'})
gnuplot.title('L2 Distance un-Normalized')

--normalize
function normalize(t)
    t:cdiv(t:norm(2,1):expandAs(t))
    return t
end


--plot normalized invariance
gnuplot.figure(3)
invariance_1 = normalize(invariance_1)
invariance_2 = normalize(invariance_2)
invariance_3 = normalize(invariance_3)
invariance_4 = normalize(invariance_4)


tick = torch.range(1,11,1)
gnuplot.plot({'TwoLayer',invariance_1,'+-'},
    {'OneLayer', invariance_2,'+-'},
    {'Linear', invariance_3,'+-'},
    {'Linear L1-regularization', invariance_4,'+-'})

gnuplot.title('L2 Distance Normalized')

--plot unnormalized invariance
invariance_1 = getDiff(model_1)
invariance_2 = getDiff(model_2)
invariance_3 = getDiff(model_3)
invariance_4 = getDiff(model_4)
gnuplot.figure(4)
gnuplot.plot({'TwoLayer',invariance_1,'+-'},
    {'OneLayer', invariance_2,'+-'},
    {'Linear', invariance_3,'+-'},
    {'Linear L1-regularization', invariance_4,'+-'})
gnuplot.title('L1 Distance un-Normalized')

--plot normalized invariance
gnuplot.figure(5)
invariance_1 = normalize(invariance_1)
invariance_2 = normalize(invariance_2)
invariance_3 = normalize(invariance_3)
invariance_4 = normalize(invariance_4)

tick = torch.range(1,11,1)
gnuplot.plot({'TwoLayer',invariance_1,'+-'},
    {'OneLayer', invariance_2,'+-'},
    {'Linear', invariance_3,'+-'},
    {'Linear L1-regularization', invariance_4,'+-'})
gnuplot.title('L1 Distance Normalized')

io.read("*n")