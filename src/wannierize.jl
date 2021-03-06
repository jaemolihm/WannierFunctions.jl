using Optim

export run_wannier_minimization

function obj(p, X, Y)
    A = X_Y_to_A(p, X, Y)

    res = spreads_MV1997(p, A, true)
    func = res.spreads.Ω
    grad = res.gradient

    gradX = zero(X)
    gradY = zero(Y)
    @views for ik = 1:p.nktot
        l_frozen = p.l_frozen[ik]
        l_not_frozen = p.l_not_frozen[ik]
        lnf = count(l_frozen)
        gradX[:,:, ik] .= Y[:,:, ik]' * grad[:,:, ik]
        gradY[:,:, ik] .= grad[:,:, ik] * X[:,:, ik]'

        gradY[l_frozen,:, ik] .= 0
        gradY[:,1:lnf, ik] .= 0
    end
    func, gradX, gradY, res
end

"""
- `f_tol``: tolerance on spread
- `g_tol``: tolerance on gradient
- `max_iter=100`: maximum optimization iterations
- `bfgs_history=20`: history size of BFGS
"""
function run_wannier_minimization(p, A; verbose=true, bfgs_history=20, max_iter=100, f_tol, g_tol)
    # initial X,Y
    X0, Y0 = A_to_XY(p, A)

    M = p.nwannier*p.nwannier + p.nband*p.nwannier
    XY0 = zeros(ComplexF64, M, p.nktot)
    for ik = 1:p.nktot
        XY0[:, ik] = vcat(vec(X0[:,:, ik]),vec(Y0[:,:, ik]))
    end

    # We have three formats:
    # (X,Y): Ntot x nw x nw, Ntot x nb x nw
    # A: ntot x nb x nw
    # XY: (nw*nw + nb*nw) x Ntot
    function fg!(G, XY)
        @assert size(G) == size(XY)
        X, Y = XY_to_X_Y(p, XY)

        f, gradX, gradY, res = obj(p, X, Y)

        @views for ik = 1:p.nktot
            G[:, ik] = vcat(vec(gradX[:,:, ik]), vec(gradY[:,:, ik]))
        end
        return f
    end

    f(XY) = fg!(similar(XY), XY)
    g!(g, XY) = (fg!(g, XY); return g)

    # need QR orthogonalization rather than SVD to preserve the sparsity structure of Y
    XYkManif = Optim.ProductManifold(Optim.Stiefel_SVD(), Optim.Stiefel_SVD(), (p.nwannier, p.nwannier), (p.nband, p.nwannier))
    XYManif = Optim.PowerManifold(XYkManif, (M,), (p.nktot,))

    ls = Optim.HagerZhang()
    meth = Optim.LBFGS

    res = Optim.optimize(f, g!, XY0,
        meth(manifold=XYManif, linesearch=ls, m=bfgs_history),
        Optim.Options(; show_trace=verbose, iterations=max_iter, f_tol, g_tol,
                      allow_f_increases=true)
    )
    verbose && display(res)
    XYmin = Optim.minimizer(res)

    Xmin, Ymin = XY_to_X_Y(p, XYmin)
    Amin = X_Y_to_A(p, Xmin, Ymin)
end