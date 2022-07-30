# We have three formats:
# (X, Y): nktot × nw × nw, nktot × nb × nw
# A: nktot × nb × nw
# XY: (nw * nw + nb * nw) × nktot

# Constraint on Y at each k point:
# Y = [I  0;  <- bands inside frozen window (I is square)
#      0  Y;  <- bands inside outer window, outside frozen window
#      0  0]  <- bands outside outer window
# Y[l_frozen, 1:length(l_frozen)] = I
# Y[l_frozen, length(l_frozen)+1:end] = 0
# Y[!l_frozen, 1:length(l_frozen)] = 0
# Y[!l_outer, :] = 0

"""
normalize and freeze a block of a matrix
A = [Uf* Ur*]
A*A = I, Uf* Uf = I
UfUf* + UrUr* = I
From this, obtain Uf*Ur = 0
Strategy: first orthogonalize Uf, then project Uf out of Ur, then orthogonalize the range of Ur
"""
function normalize_and_freeze(A,frozen,not_frozen_outer)
    # orthogonalize Uf
    Uf = A[frozen,:]'
    U,S,V = svd(Uf)
    Uf = U*V'
    # Uf = normalize_matrix_chol(Uf)

    # project Uf out of Ur
    Ur = A[not_frozen_outer,:]'
    Ur -= Uf*Uf'*Ur

    # # alternative method, maybe more stable but slower
    # ovl = Ur'Ur
    # S, U = eig(Hermitian(ovl))
    # S = real(S)
    # @assert !any(x -> 1e-11 <= x <= 1e-9, S)
    # @assert count(x -> x > 1e-10, S) == size(A,2) - nfrozen
    # Sm12 = map(x-> x < 1e-10 ? 0. : 1/sqrt(x), S)
    # Ur = Ur*(U*diagm(Sm12)*U')

    # renormalize the range of Ur
    U,S,V = svd(Ur)
    eps = 1e-10
    @assert !any(x -> 1e-11 <= x <= 1e-9, S)
    @assert count(x -> x > 1e-10, S) == size(A,2) - count(frozen)
    S[S .> eps] .= 1
    S[S .< eps] .= 0
    Ur = U*Diagonal(S)*V'

    A[not_frozen_outer,:] = Ur'

    B = zero(A)
    B[frozen,:] .= Uf'
    B[not_frozen_outer,:] .= Ur'
    @assert isapprox(B'B, I, rtol=1e-12)
    @assert isapprox(B[frozen,:]*B[frozen,:]', I, rtol=1e-12)
    @assert norm(Uf'*Ur)<1e-10
    return B
end

function _check_X_Y(X, Y, l_frozen, l_not_frozen, l_outer)
    lnf = count(l_frozen)
    @assert Y' * Y ≈ I
    @assert X' * X ≈ I
    @assert Y[l_frozen, 1:lnf] ≈ I
    @assert all(Y[l_not_frozen, 1:lnf] .≈ 0)
    @assert all(Y[l_frozen, lnf+1:end] .≈ 0)
    @assert all(Y[.!l_outer, :] .≈ 0)
end

function X_Y_to_A(p, X, Y)
    A = zeros(ComplexF64, p.nband, p.nwannier, p.nktot)
    @views for ik = 1:p.nktot
        l_frozen = p.l_frozen[ik]
        l_not_frozen = p.l_not_frozen[ik]
        l_outer = p.l_outer[ik]
        _check_X_Y(X[:, :, ik], Y[:, :, ik], l_frozen, l_not_frozen, l_outer)
        mul!(A[:, :, ik], Y[:, :, ik], X[:, :, ik])
    end
    A
end

function A_to_XY(p,A)
    X = zeros(ComplexF64, p.nwannier, p.nwannier, p.nktot)
    Y = zeros(ComplexF64, p.nband, p.nwannier, p.nktot)
    @views for ik = 1:p.nktot
        l_frozen = p.l_frozen[ik]
        l_not_frozen = p.l_not_frozen[ik]
        l_outer = p.l_outer[ik]
        lnf = count(l_frozen)

        l_not_frozen_outer = l_not_frozen .& l_outer
        Afrozen = normalize_and_freeze(A[:,:, ik], l_frozen, l_not_frozen_outer)
        Af = Afrozen[l_frozen,:]
        Ar = Afrozen[l_not_frozen_outer,:]

        #determine Y
        if lnf != p.nwannier
            proj = Ar*Ar'
            proj = Hermitian((proj+proj')/2)
            D,V = eigen(proj) #sorted by increasing eigenvalue
        end
        Y[l_frozen, 1:lnf, ik] = I(lnf)
        if lnf != p.nwannier
            Y[l_not_frozen_outer,lnf+1:end, ik] = V[:,end-p.nwannier+lnf+1:end]
        end

        #determine X
        Xleft, S, Xright = svd(Y[:,:, ik]'*Afrozen)
        X[:,:, ik] = Xleft*Xright'

        _check_X_Y(X[:, :, ik], Y[:, :, ik], l_frozen, l_not_frozen, l_outer)
        @assert Y[:,:, ik]*X[:,:, ik] ≈ Afrozen
    end
    X, Y
end

"convert XY to (X, Y)"
function XY_to_X_Y(p, XY)
    X = zeros(ComplexF64, p.nwannier, p.nwannier, p.nktot)
    Y = zeros(ComplexF64, p.nband, p.nwannier, p.nktot)
    @views for ik = 1:p.nktot
        XYk = XY[:, ik]
        X[:, :, ik] .= reshape(XYk[1:p.nwannier*p.nwannier], (p.nwannier, p.nwannier))
        Y[:, :, ik] .= reshape(XYk[p.nwannier*p.nwannier+1:end], (p.nband, p.nwannier))
    end
    X, Y
end
