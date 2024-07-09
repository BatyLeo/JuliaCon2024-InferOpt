using InferOpt
using Plots
using LaTeXStrings
using Colors
using LinearAlgebra

logocolors = Colors.JULIA_LOGO_COLORS

function get_angle(v)
    @assert !(norm(v) ≈ 0)
    v = v ./ norm(v)
    if v[2] >= 0
        return acos(v[1])
    else
        return π + acos(-v[1])
    end
end;

function init_plot(title)
    pl = plot(;
        aspect_ratio=:equal,
        legend=:outerleft,
        xlim=(-1.1, 1.1),
        ylim=(-1.1, 1.1),
        title=title,
    )
    return pl
end;

function plot_polytope!(pl, vertices)
    plot!(
        vcat(map(first, vertices), first(vertices[1])),
        vcat(map(last, vertices), last(vertices[1]));
        fillrange=0,
        fillcolor=:gray,
        fillalpha=0.2,
        linecolor=:black,
        label=L"\mathrm{conv}(\mathcal{V})",
    )
end;

function plot_objective!(pl, θ)
    plot!(
        pl,
        [0.0, θ[1]],
        [0.0, θ[2]],
        color=logocolors.purple,
        arrow=true,
        lw=2,
        label=nothing,
    )
    Plots.annotate!(pl, [-0.2 * θ[1]], [-0.2 * θ[2]], [L"\theta"])
    return θ
end;

function plot_maximizer!(pl, θ, polytope, maximizer)
    ŷ = maximizer(θ; polytope)
    scatter!(
        pl,
        [ŷ[1]],
        [ŷ[2]];
        color=logocolors.red,
        markersize=9,
        markershape=:square,
        label=L"f(\theta)",
    )
end;

# ╔═╡ 269547da-f4ec-4746-9453-5cb8d7703da8
function plot_distribution!(pl, probadist)
    A = probadist.atoms
    As = sort(A, by=get_angle)
    p = probadist.weights
    plot!(
        pl,
        vcat(map(first, As), first(As[1])),
        vcat(map(last, As), last(As[1]));
        fillrange=0,
        fillcolor=:blue,
        fillalpha=0.1,
        linestyle=:dash,
        linecolor=logocolors.blue,
        label=L"\mathrm{conv}(\hat{p}(\theta))",
    )
    scatter!(
        pl,
        map(first, A),
        map(last, A);
        markersize=25 .* p .^ 0.5,
        markercolor=logocolors.blue,
        markerstrokewidth=0,
        markeralpha=0.4,
        label=L"\hat{p}(\theta)",
    )
end;

function plot_expectation!(pl, probadist)
    ŷΩ = compute_expectation(probadist)
    scatter!(
        pl,
        [ŷΩ[1]],
        [ŷΩ[2]];
        color=logocolors.blue,
        markersize=6,
        markershape=:hexagon,
        label=L"\hat{f}(\theta)",
    )
end;

function compress_distribution!(
    probadist::FixedAtomsProbabilityDistribution{A,W};
    atol=0,
) where {A,W}
    (; atoms, weights) = probadist
    to_delete = Int[]
    for i in length(probadist):-1:1
        ai = atoms[i]
        for j in 1:(i-1)
            aj = atoms[j]
            if isapprox(ai, aj; atol=atol)
                weights[j] += weights[i]
                push!(to_delete, i)
                break
            end
        end
    end
    sort!(to_delete)
    deleteat!(atoms, to_delete)
    deleteat!(weights, to_delete)
    return probadist
end;

function main(N=7; α=0.5 * π)
    N = 7
    polytope = [[cospi(2k / N), sinpi(2k / N)] for k in 0:N-1]
    maximizer(θ; polytope) = polytope[argmax(dot(θ, v) for v in polytope)]

    angle_oracle = α
    θ = 0.5 .* [cos(angle_oracle), sin(angle_oracle)]
    pl = init_plot("Linear oracle")
    plot_polytope!(pl, polytope)
    plot_objective!(pl, θ)
    plot_maximizer!(pl, θ, polytope, maximizer)
    return pl
end

function main_perturbed(N=7; α=0.5 * π, plot_probadist_perturbed=true)
    N = 7
    polytope = [[cospi(2k / N), sinpi(2k / N)] for k in 0:N-1]
    maximizer(θ; polytope) = polytope[argmax(dot(θ, v) for v in polytope)]
    perturbed_layer = PerturbedAdditive(maximizer; ε=0.2, nb_samples=100, seed=0)

    angle_perturbed = α
    θ = 0.5 .* [cos(angle_perturbed), sin(angle_perturbed)]
    probadist = compute_probability_distribution(perturbed_layer, θ; polytope)
    compress_distribution!(probadist)
    pl = init_plot("Perturbation")
    plot_polytope!(pl, polytope)
    plot_objective!(pl, θ)
    plot_probadist_perturbed && plot_distribution!(pl, probadist)
    plot_maximizer!(pl, θ, polytope, maximizer)
    plot_expectation!(pl, probadist)
end

main()
main_perturbed()
