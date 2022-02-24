module NN

using Flux
using Flux: onehotbatch, onecold
using MLDatasets
using Random
using CUDA
using BSON: @save, @load


export trainmodel
export accuracy
export getprocessedMNIST
export getprocessedFashionMNIST


function getprocessedMNIST()
	train_imgs, train_labels = MLDatasets.MNIST.traindata(Float32)
	test_imgs, test_labels = MLDatasets.MNIST.testdata(Float32)

    # Reshape from (width, height, samples) to (width, height, channels (1), samples)
    train_imgs = Flux.unsqueeze(train_imgs, ndims(train_imgs))
    test_imgs = Flux.unsqueeze(test_imgs, ndims(test_imgs))

	train_labels = Float32.(onehotbatch(train_labels, 0:9))
    test_labels = Float32.(onehotbatch(test_labels, 0:9))

    return (
        train=(
            inputs=train_imgs,
            labels=train_labels
        ),
        test=(
            inputs=test_imgs,
            labels=test_labels
        )
    )
end

function getprocessedFashionMNIST()
	train_imgs, train_labels = MLDatasets.FashionMNIST.traindata(Float32)
	test_imgs, test_labels = MLDatasets.FashionMNIST.testdata(Float32)

    # Reshape from (width, height, samples) to (width, height, channels (1), samples)
    train_imgs = Flux.unsqueeze(train_imgs, ndims(train_imgs))
    test_imgs = Flux.unsqueeze(test_imgs, ndims(test_imgs))

	train_labels = Float32.(onehotbatch(train_labels, 0:9))
    test_labels = Float32.(onehotbatch(test_labels, 0:9))

    return (
        train=(
            inputs=train_imgs,
            labels=train_labels
        ),
        test=(
            inputs=test_imgs,
            labels=test_labels
        )
    )
end

percstr(x::Real; digits=2) = "$(round(100x; digits=digits))%"
accuracy(model, x, y) = sum(onecold(cpu(model(x))) .== onecold(cpu(y))) / size(x)[end]

function trainmodel(;
    name::String,
    model,
    dataset,
    loss,
    opt,
    batchsize::Int,
    epochs::Int,
    device = gpu,
)
    savefilename = "saves/$name.bson"
    # Check for preexisting trained weights
    if (isfile(savefilename))
        @info "Found a preexisting trained model in file '$savefilename'. Loading from it..."
        @load savefilename parameters test_acc
        model = cpu(model)
        try
            Flux.loadparams!(model, parameters)
            @info "Model loaded. Accuracy: $(percstr(test_acc))"
            return device(model)
        catch e
            @warn "Current model is not compatible with the weights in file '$savefilename'"
            @info model
            @info "Training from scratch..."
            @warn "File '$savefilename' will be overwritten after training"
        end
    end
    @info "Performing experiment `$name`..."
    @info "Loading model and dataset to $((device == gpu) && has_cuda() ? "GPU" : "CPU")..."
    dataset = device(dataset)
    model = device(model)

    train_dataloader = Flux.Data.DataLoader((dataset.train.inputs, dataset.train.labels), batchsize=batchsize, shuffle=false)

    loss′(x, y) = loss(model(x), y)

    @info "Compiling model and calculating initial accuracy..."
    # Keep track of best results so far so we can roll back to it at the end of training
    best_acc = accuracy(model, dataset.test.inputs, dataset.test.labels)    # The test set is used here for debug purposes. This should be overwritten soon
    best_params = deepcopy(params(model))
    @info "Initial test accuracy: $(percstr(best_acc))"

    avgsecsperepoch = 0f0

    @info "Training..."
    # Those flushes increase responsiveness, specially if `stderr` is redirected to a file
    flush(stderr)
    for epoch_idx in 1:epochs
        # Train for a single epoch
        secsthisepoch = @elapsed Flux.train!(loss′, params(model), train_dataloader, opt)

        # The first epoch time is discarded as it includes (large) compilation times
        if epoch_idx > 1
            avgsecsperepoch += (secsthisepoch - avgsecsperepoch) / (epoch_idx - 1)
        end

        train_acc = accuracy(model, dataset.train.inputs, dataset.train.labels)

        # Log progress highlighting in red new best result
        @info "$name [$epoch_idx/$epochs] Train Accuracy: $(percstr(train_acc))\t($(round(avgsecsperepoch; digits=2)) secs/epoch)$(train_acc > best_acc ? "\t\e[91mNew best acc\e[m" : "")"

        # Register the best accuracy so far
        if train_acc > best_acc
            best_acc = train_acc
            best_params = deepcopy(params(model))
        end

        flush(stderr)
    end

    @info "Time spent training: $(round(avgsecsperepoch*epochs; digits=2)) secs"

    # Rewind weights to the best performing found during training
    Flux.loadparams!(model, best_params)
    test_acc = accuracy(model, dataset.test.inputs, dataset.test.labels)
    @save savefilename parameters=params(cpu(model)) test_acc=test_acc
    @info "Test accuracy: $(percstr(test_acc))"

    return model
end

end
