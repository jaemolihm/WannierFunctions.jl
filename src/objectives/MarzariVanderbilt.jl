using Printf
using LinearAlgebra

export MarzariVanderbiltObjective

imaglog(z) = atan(imag(z), real(z))

struct SpreadResult{T}
    nwannier::Int
    centers::Vector{SVector{3, T}}
    spreads::Vector{T}
    Ω::T
    ΩI::T
    ΩD::T
    ΩOD::T
end

function Base.show(io::IO, res::SpreadResult)
    print(io, "Wannier centers and spreads (Angstrom)\n")
    for iw in 1:res.nwannier
        print(io, @sprintf "%d %15.8f %15.8f %15.8f %15.8f\n" iw res.centers[iw][1] res.centers[iw][2] res.centers[iw][3] res.spreads[iw])
    end
    print(io, "Sum of centers and spreads $(sum(res.centers)) $(sum(res.spreads))\n")
    print(io, "Omega I     = $(res.ΩI)\n")
    print(io, "Omega D     = $(res.ΩD)\n")
    print(io, "Omega OD    = $(res.ΩOD)\n")
    print(io, "Omega Total = $(res.Ω)\n")
end

"""
Marzari and Vanderbilt (1997)
"""
Base.@kwdef struct MarzariVanderbiltObjective <: AbstractWannierObjective
    nband::Int
    nwannier::Int
    nktot::Int
    nnb::Int
    ik_rng::UnitRange{Int} = 1:nktot
    neighbors::Matrix{Int}
    wbs::Vector{Float64}
    bvecs_cart::Vector{SVector{3, Float64}}
    mmn::Array{ComplexF64, 4}
    # Preallocated buffers
    Rkb::Matrix{ComplexF64} = zeros(ComplexF64, nband, nwannier)
    Tkb::Matrix{ComplexF64} = zeros(ComplexF64, nband, nwannier)
    M::Array{ComplexF64, 4} = zeros(ComplexF64, nwannier, nwannier, nnb, nktot)
    O::Array{ComplexF64, 4} = zeros(ComplexF64, nband, nwannier, nnb, nktot)
end

function Base.show(io::IO, obj::MarzariVanderbiltObjective)
    print(io, "MarzariVanderbiltObjective(")
    print(io, "nband = ", obj.nband)
    print(io, ", nwannier = ", obj.nwannier)
    print(io, ", nktot = ", obj.nktot)
    print(io, ", nnb = ", obj.nnb)
    print(io, ")")
end

function compute_objective_and_gradient!(gradient, U, obj::MarzariVanderbiltObjective)
    r = zeros(SVector{3, Float64}, obj.nwannier)
    r2 = zeros(obj.nwannier)
    ΩI = 0.
    ΩOD = 0.
    ΩD = 0.
    Rkb = obj.Rkb
    Tkb = obj.Tkb

    # Mkb = U(k)<uk|ukb>U(kb) (mmn in Wannier gauge)
    # Okb = <uk|ukb>U(kb)

    for ik in obj.ik_rng
        @views for ib in 1:obj.nnb
            ikb = obj.neighbors[ib, ik]
            wb = obj.wbs[ib]
            b = obj.bvecs_cart[ib]
            Okb = view(obj.O, :, :, ib, ik)
            Mkb = view(obj.M, :, :, ib, ik)
            mul!(Okb, obj.mmn[:, :, ib, ik], U[:, :, ikb])
            mul!(Mkb, U[:, :, ik]', Okb)

            # Compute centers and spreads
            for n = 1:obj.nwannier
                # Eq. (31) of MV1997
                r[n] -= wb * b * imaglog(Mkb[n, n])
                r2[n] += wb * (1 - abs(Mkb[n, n])^2 + imaglog(Mkb[n, n])^2)
            end
            ΩI += wb * (obj.nwannier - sum(abs2, Mkb))
            ΩOD += wb * (sum(abs2, Mkb) - sum(abs2, Diagonal(Mkb)))
        end
    end
    r ./= obj.nktot
    r2 ./= obj.nktot
    ΩI /= obj.nktot
    ΩOD /= obj.nktot
    spreads = @. r2 - norm(r)^2
    Ωtot = sum(spreads)
    ΩD = Ωtot - ΩI - ΩOD
    spread_result = SpreadResult(obj.nwannier, r, spreads, Ωtot, ΩI, ΩD, ΩOD)

    if gradient !== nothing
        for ik in obj.ik_rng
            @views for ib in 1:obj.nnb
                wb = obj.wbs[ib]
                b = obj.bvecs_cart[ib]

                Okb = view(obj.O, :, :, ib, ik)
                Mkb = view(obj.M, :, :, ib, ik)

                # MV way
                # A(B) = (B-B')/2
                # S(B) = (B+B')/(2*im)
                # q = imaglog.(diag(Mkb)) + centers'*b
                # for m=1:obj.nwannier,n=1:obj.nwannier
                #     R[m,n] = Mkb[m,n]*conj(Mkb[n,n])
                #     T[m,n] = Mkb[m,n]/Mkb[n,n]*q[n]
                # end
                # gradient[:, :, ik] += 4*obj.wbs*(A(R) .- S(T))

                for n = 1:obj.nwannier
                    if abs(Mkb[n, n]) < 1e-10
                        # error if division by zero. Should not happen if the initial gauge
                        # is not too bad
                        println("Mkbnn too large! $n $ib $ik")
                        display(Mkb)
                        error()
                    end

                    # Eq.(47) of MV1997
                    q = imaglog(Mkb[n, n]) + b' * r[n]
                    Tfac = -im * q / Mkb[n, n]
                    for m in 1:obj.nband
                        Rkb[m, n] = -Okb[m, n] * conj(Mkb[n, n])
                        Tkb[m, n] = Tfac * Okb[m, n]
                    end
                end
                @. gradient[:, :, ik] += 4 * wb * (Rkb + Tkb)
            end
        end
        gradient ./= obj.nktot
    end

    if gradient !== nothing
        (; objective=Ωtot, gradient, spreads=spread_result)
    else
        (; objective=Ωtot, spreads=spread_result)
    end
end
