using LinearAlgebra

export spreads_MV1997

imaglog(z) = atan(imag(z), real(z))

struct SpreadResults_X3X{T}
    nwannier::Int
    centers::Vector{SVector{3, T}}
    spreads::Vector{T}
    Ω::T
    ΩI::T
    ΩD::T
    ΩOD::T
end

function Base.show(io::IO, res::SpreadResults_X3X)
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

function spreads_MV1997(p, U, compute_grad=true)
    grad = zeros(ComplexF64, p.nband, p.nwannier, p.nktot)
    r = zeros(SVector{3, Float64}, p.nwannier)
    r2 = zeros(p.nwannier)
    ΩI = 0.
    ΩOD = 0.
    ΩD = 0.
    # frozen_weight = 0.
    Rkb = zeros(ComplexF64, p.nband, p.nwannier)
    Tkb = zeros(ComplexF64, p.nband, p.nwannier)
    M = zeros(ComplexF64, p.nwannier, p.nwannier, p.nnb, p.nktot)
    O = zeros(ComplexF64, p.nband, p.nwannier, p.nnb, p.nktot)

    for ik = 1:p.nktot
        @views for ib = 1:p.nnb
            ikb = p.neighbors[ib, ik]
            wb = p.wbs[ib]
            b = p.bvecs_cart[ib]
            Okb = view(O, :, :, ib, ik)
            Mkb = view(M, :, :, ib, ik)
            mul!(Okb, p.M_bands[:, :, ib, ik], U[:, :, ikb])
            mul!(Mkb, U[:, :, ik]', Okb)

            # Compute centers and spreads
            for n = 1:p.nwannier
                # Eq. (31) of MV1997
                r[n] -= wb * b * imaglog(Mkb[n, n])
                r2[n] += wb *(1-abs(Mkb[n, n])^2 + imaglog(Mkb[n, n])^2)
            end
        end
    end
    r ./= p.nktot
    r2 ./= p.nktot

    # Levitt <-> Wannier90
    # A          U (full, nb x nw semi-unitary matrix)
    # Mkb        U(k)<uk|ukb>U(kb) (mmn in Wannier gauge)
    # Okb        <uk|ukb>U(kb)

    for ik = 1:p.nktot
        @views for ib = 1:p.nnb
            ikb = p.neighbors[ib, ik]
            wb = p.wbs[ib]
            b = p.bvecs_cart[ib]

            Okb = view(O, :, :, ib, ik)
            Mkb = view(M, :, :, ib, ik)

            if compute_grad
                # #MV way
                # A(B) = (B-B')/2
                # S(B) = (B+B')/(2*im)
                # q = imaglog.(diag(Mkb)) + centers'*b
                # for m=1:p.nwannier,n=1:p.nwannier
                #     R[m,n] = Mkb[m,n]*conj(Mkb[n,n])
                #     T[m,n] = Mkb[m,n]/Mkb[n,n]*q[n]
                # end
                # grad[:, :, ik] += 4*p.wbs*(A(R) .- S(T))

                for n = 1:p.nwannier
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
                    for m = 1:p.nband
                        Rkb[m, n] = -Okb[m, n] * conj(Mkb[n, n])
                        Tkb[m, n] = Tfac * Okb[m, n]
                    end
                end
                @. grad[:, :, ik] += 4 * wb * (Rkb + Tkb)
            end

            ΩI += wb * (p.nwannier - sum(abs2, Mkb))
            ΩOD += wb * (sum(abs2, Mkb) - sum(abs2, Diagonal(Mkb)))
        end
    end
    ΩI /= p.nktot
    ΩOD /= p.nktot
    # frozen_weight /= Ntot
    grad ./= p.nktot

    spreads = @. r2 - norm(r)^2
    Ωtot = sum(spreads)
    ΩD = Ωtot - ΩI - ΩOD
    # return Omega_res(Ωtot,ΩI,ΩOD,ΩD,Ωtilde,frozen_weight,spreads,r,grad)
    (; spreads=SpreadResults_X3X(p.nwannier, r, spreads, Ωtot, ΩI, ΩD, ΩOD), gradient=grad)
end
