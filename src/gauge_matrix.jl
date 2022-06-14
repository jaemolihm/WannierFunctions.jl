# We have three formats:
# (X,Y): Ntot x nw x nw, Ntot x nb x nw
# A: ntot x nb x nw
# XY: (nw*nw + nb*nw) x Ntot

"""
normalize and freeze a block of a matrix
A = [Uf* Ur*]
A*A = I, Uf* Uf = I
UfUf* + UrUr* = I
From this, obtain Uf*Ur = 0
Strategy: first orthogonalize Uf, then project Uf out of Ur, then orthogonalize the range of Ur
"""
function normalize_and_freeze(A,frozen,not_frozen)
    # orthogonalize Uf
    Uf = A[frozen,:]'
    U,S,V = svd(Uf)
    Uf = U*V'
    # Uf = normalize_matrix_chol(Uf)

    # project Uf out of Ur
    Ur = A[not_frozen,:]'
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

    A[not_frozen,:] = Ur'


    B = vcat(Uf',Ur')
    B[frozen,:] .= Uf'
    B[not_frozen,:] .= Ur'
    @assert isapprox(B'B, I, rtol=1e-12)
    @assert isapprox(B[frozen,:]*B[frozen,:]', I, rtol=1e-12)
    @assert norm(Uf'*Ur)<1e-10
    return B
end



function XY_to_A(p, X, Y)
    A = zeros(ComplexF64, p.nband, p.nwannier, p.nktot)
    @views for ik = 1:p.nktot
        # l_frozen, l_not_frozen = local_frozen_sets(p, nfrozen, i, j, k, frozen_window_low, frozen_window_high)
        l_frozen = 1:0
        l_not_frozen = 1:p.nband
        lnf = count(l_frozen)
        @assert Y[:,:, ik]'Y[:,:, ik] ≈ I
        @assert X[:,:, ik]'X[:,:, ik] ≈ I
        @assert Y[l_frozen,1:lnf, ik] ≈ I
        @assert norm(Y[l_not_frozen,1:lnf, ik]) ≈ 0
        @assert norm(Y[l_frozen,lnf+1:end, ik]) ≈ 0
        A[:, :, ik] .= Y[:, :, ik] * X[:, :, ik]
        # @assert normalize_and_freeze(A[:,:, ik],lnf) ≈ A[:,:, ik] rtol=1e-4
    end
    A
end

function A_to_XY(p,A)
    X = zeros(ComplexF64, p.nwannier, p.nwannier, p.nktot)
    Y = zeros(ComplexF64, p.nband, p.nwannier, p.nktot)
    @views for ik = 1:p.nktot
        # l_frozen, l_not_frozen = local_frozen_sets(p, nfrozen, i, j, k, frozen_window_low, frozen_window_high)
        l_frozen = 1:0
        l_not_frozen = 1:p.nband

        lnf = count(l_frozen)
        Afrozen = normalize_and_freeze(A[:,:, ik], l_frozen, l_not_frozen)
        Af = Afrozen[l_frozen,:]
        Ar = Afrozen[l_not_frozen,:]

        #determine Y
        if lnf != p.nwannier
            proj = Ar*Ar'
            proj = Hermitian((proj+proj')/2)
            D,V = eigen(proj) #sorted by increasing eigenvalue
        end
        Y[l_frozen,1:lnf, ik] = Matrix(I,lnf,lnf)
        if lnf != p.nwannier
            Y[l_not_frozen,lnf+1:end, ik] = V[:,end-p.nwannier+lnf+1:end]
        end

        #determine X
        Xleft, S, Xright = svd(Y[:,:, ik]'*Afrozen)
        X[:,:, ik] = Xleft*Xright'

        @assert Y[:,:, ik]'Y[:,:, ik] ≈ I
        @assert X[:,:, ik]'X[:,:, ik] ≈ I
        @assert Y[l_frozen,1:lnf, ik] ≈ I
        @assert norm(Y[l_not_frozen,1:lnf, ik]) ≈ 0
        @assert norm(Y[l_frozen,lnf+1:end, ik]) ≈ 0
        @assert Y[:,:, ik]*X[:,:, ik] ≈ Afrozen
    end
    X, Y
end

"convert XY to (X, Y)"
function XY_to_XY(p, XY)
    X = zeros(ComplexF64, p.nwannier, p.nwannier, p.nktot)
    Y = zeros(ComplexF64, p.nband, p.nwannier, p.nktot)
    for ik = 1:p.nktot
        XYk = XY[:, ik]
        X[:,:, ik] = reshape(XYk[1:p.nwannier*p.nwannier], (p.nwannier, p.nwannier))
        Y[:,:, ik] = reshape(XYk[p.nwannier*p.nwannier+1:end], (p.nband, p.nwannier))
    end
    X, Y
end