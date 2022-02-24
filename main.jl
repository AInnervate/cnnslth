using Random: seed!
seed!(4231)
using Flux
using Zygote
using BSON: @save, @load
ENV["PLOTS_DEFAULT_BACKEND"] = "PyPlot"
using Plots
pyplot()
using StatsPlots

include("NN.jl")
using .NN
include("Solver.jl")
using .Solver


function RSSapproximate(model; ε::Real, n::Real, hoursperlayer::Real = 0, experimentname::String)
    @assert 0 < ε
    @assert n > 0

    # Make sure the original model is not altered and the copy is in CPU
    model = cpu(deepcopy(model))
    filename = "saves/$experimentname-RSS-n$n-h$hoursperlayer.bson"
    # Check for cached model
    if isfile(filename)
        @info "Found a saved approximation in file '$filename'. Loading from it..."
        @load filename parameters
        try
            Flux.loadparams!(model, parameters)
        catch e
            @warn "Current model is not compatible with the weights in file '$filename'"
            @info "$model\nReturning"
            return nothing
        end
    else
        nparams = countparams(model)
        @info "Subset Summing $nparams parameters\n  n = $n\n  ε = $(ε/nparams)"
        for l ∈ model
            if l isa Dense || l isa Conv
                @info "Subset Summing $l layer with $(sum(length, params(l))) parameters"
                for p ∈ params(l)
                    @time p .= randomsubsetsum.(p, ε = ε/nparams, n = n, verbose = false, minsecs = hoursperlayer*60*60/nparams)
                end
            elseif l isa MaxPool || l isa typeof(flatten)
                # Nothing to be done
            else
                @warn "TODO: Support layer of type `$(typeof(l))`"
                @info "Ignoring"
            end
        end
        @save filename parameters=params(cpu(model))
    end
    return model
end

function buildlenet5(; inputsize::Tuple{Int, Int, Int}, nclasses::Int, init::Function = Flux.kaiming_uniform)
    cnn = Chain(
        Conv((5, 5), inputsize[end]=>6, pad = SamePad(), relu, init = init),
        MaxPool((2,2), stride=2),

        Conv((5, 5), 6=>16, pad = SamePad(), relu, init = init),
        MaxPool((2,2), stride=2),
    )

    cnnoutsize = Flux.outputsize(cnn, inputsize, padbatch=true)

    return Chain(
        cnn...,
        flatten,
        Dense(prod(cnnoutsize), 120, relu, init = init),
        Dense(120, 84, relu, init = init),
        Dense(84, nclasses, init = init),
    )
end

countparams(model) = sum(length, params(model))

function run()
    for dataparams ∈ [
        (datasetname = "MNIST", dataset = getprocessedMNIST()),
        (datasetname = "FashionMNIST", dataset = getprocessedFashionMNIST()),
    ]
        experimentname = "lenet5-$(dataparams.datasetname)"
        model = trainmodel(
            name = experimentname,
            model = buildlenet5(inputsize=(28,28,1), nclasses=10),
            dataset = dataparams.dataset,
            loss = (ŷ, y)->Flux.logitcrossentropy(ŷ, y),
            opt = ADAM(0.001),
            epochs = 50,
            batchsize = 64,
        )
        println("l1 = ", map(W->round(sum(abs, W), digits = 2), params(model)))
        println("l∞ = ", map(W->round(maximum(abs, W), digits = 2), params(model)))
        model = cpu(model[1:4])  # Just the convolutional layers
        ns = 5:5:30
        # Large ε is effectively ignored
        approxmodels = [n => RSSapproximate(model, ε = 10*countparams(model), n = n, hoursperlayer = Inf, experimentname = experimentname) for n ∈ ns]

        maxweightdeltas = [
            map(params(model), params(modelRSS)) do p, p′
                maximum(abs, (p .- p′) ./ maximum(abs, p))
            end
            for modelRSS ∈ last.(approxmodels)
        ]
        plot(
            first.(approxmodels),
            maximum.(maxweightdeltas),
            legends = false,
            marker = :circle,
            xlabel = "sample size",
            ylabel = "maximum relative weight error",
            yticks = 10.0 .^ (0:-1:-4),
            yaxis = :log,
            size = (600, 350),
        )
        savefig("plots/maxweightdelta-$(dataparams.datasetname).pdf")

        @info "Computing outputs"
        # Use GPU if available
        model = gpu(model)
        approxmodels = @. first(approxmodels) => gpu(last(approxmodels))
        inputs_train = gpu(dataparams.dataset.train.inputs)
        inputs_test = gpu(dataparams.dataset.test.inputs)
        referenceoutputs_train = model(inputs_train)
        referenceoutputs_test = model(inputs_test)
        outputerrors_train = [cpu(abs.(referenceoutputs_train .- modelRSS(inputs_train)) ./ maximum(abs, referenceoutputs_train)) for modelRSS ∈ last.(approxmodels)]
        outputerrors_test = [cpu(abs.(referenceoutputs_test .- modelRSS(inputs_test)) ./ maximum(abs, referenceoutputs_test)) for modelRSS ∈ last.(approxmodels)]
        plot(
            first.(approxmodels),
            maximum.(outputerrors_train),
            label = "train",
            marker = :circle,
            line = :solid,
        )
        plot!(
            first.(approxmodels),
            maximum.(outputerrors_test),
            label = "test",
            marker = :circle,
            line = :dash,
        )
        plot!(size = (600, 350))
        xlabel!("sample size")
        ylabel!("maximum relative output error")
        yaxis!(:log)
        yticks!(10.0 .^ (0:-1:-5))
        savefig("plots/maxoutputerrors-$(dataparams.datasetname).pdf")
    end
    return nothing
end

run()
