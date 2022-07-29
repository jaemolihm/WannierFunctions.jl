using StaticArrays
using Dates
using Unitful
using UnitfulAtomic

export read_mmn, read_amn, read_eig

function read_mmn(seedname, kpts)
    println("Reading $seedname.mmn")
    file = open("$seedname.mmn")
    readline(file) # skip header
    arr = split(readline(file))
    nband, nk, nnb = parse.(Int64, arr)
    @assert nk == length(kpts)

    mmn = zeros(ComplexF64, nband, nband, nnb, nk) # overlap matrix, (ik) representation
    ik_neighbors = zeros(Int, nnb, nk) # for each point, list of ik_neighbors, (ik) representation
    displ_vecs = zeros(Int, 3, nnb, nk)
    for _ in 1:nk
        for inb = 1:nnb
            arr = split(readline(file))
            ik = parse(Int64, arr[1])
            ikb = parse(Int64, arr[2])
            @. displ_vecs[:, inb, ik] = parse(Int64, arr[3:5])
            ik_neighbors[inb, ik] = ikb
            for n = 1:nband, m = 1:nband
                arr = split(readline(file))
                mmn[m, n, inb, ik] = complex(parse.(Float64, arr)...)
            end
        end
    end

    # Make b vector ordering uniform for all k points
    bvecs = reorder_bvecs!(mmn, ik_neighbors, displ_vecs, kpts)

    mmn, bvecs, ik_neighbors
end

function reorder_bvecs!(mmn, ik_neighbors, displ_vecs, kpts)
    # Find list of b vectors according to that of ik = 1
    nnb, nk = size(ik_neighbors)
    get_bvecs(ik) = [SVector{3}(kpts[ik_neighbors[inb, ik]] - kpts[ik] .+ displ_vecs[:, inb, ik]) for inb in 1:nnb]
    bvecs = get_bvecs(1)

    # Order inb according to bvecs
    for ik = 1:nk
        bvecs_ik = get_bvecs(ik)
        inbs_new = [findfirst(x -> x ≈ b, bvecs_ik) for b in bvecs]
        @assert Set(inbs_new) == Set(1:nnb)
        mmn[:, :, :, ik] .= mmn[:, :, inbs_new, ik]
        ik_neighbors[:, ik] .= ik_neighbors[inbs_new, ik]
        displ_vecs[:, :, ik] .= displ_vecs[:, inbs_new, ik]
    end
    bvecs
end

function read_amn(seedname)
    println("Reading $seedname.amn")
    file = open("$seedname.amn")

    readline(file)
    arr = split(readline(file))
    nband, nk, nw = parse.(Int64, arr)

    amn = zeros(ComplexF64, nband, nw, nk)
    lines = readlines(file)
    for line in lines
        arr = split(line)
        m = parse(Int64, arr[1])
        n = parse(Int64, arr[2])
        ik = parse(Int64, arr[3])
        amn[m, n, ik] = complex(parse.(Float64, arr[4:5])...)
    end

    amn
end

function read_eig(seedname)
    local nband, nk
    println("Reading $seedname.eig")
    eigfile = open("$seedname.eig")
    eigs = Float64[]
    while(!eof(eigfile))
        arr = split(readline(eigfile))
        nband = parse(Int, arr[1])
        nk = parse(Int, arr[2])
        e = parse(Float64, arr[3])
        push!(eigs, e)
    end
    reshape(eigs, nband, nk)
end

function write_amn(U, seedname)
    nband, nwannier, nktot = size(U)
    f = open("$seedname.amn","w")
    write(f, "Created by WannierFunctions.jl ", string(now()), "\n")
    write(f, "$nband $nktot $nwannier\n")
    for ik = 1:nktot
        for iw in 1:nwannier
            for ib in 1:nband
                coeff = U[ib, iw, ik]
                write(f, "$ib $iw $ik $(real(coeff)) $(imag(coeff))\n")
            end
        end
    end
    close(f)
end

function read_dmn(seedname, nwannier)
    println("Reading $seedname.dmn")
    file = open("$seedname.dmn")
    readline(file) # skip header
    nband, nsymmetry, nkirr, nktot = parse.(Int, split(readline(file)))
    readline(file)

    function _read_integers(file)
        data = Int[]
        while !eof(file)
            line = readline(file)
            all(isspace, line) && break
            append!(data, parse.(Int, split(line)))
        end
        data
    end
    ik_to_ikirr = _read_integers(file)
    ikirr_to_ik = _read_integers(file)
    ikirr_isym_to_isk = [_read_integers(file) for _ in 1:nkirr]

    d_matrix_wann = zeros(ComplexF64, nwannier, nwannier, nsymmetry, nkirr)
    d_matrix_band = zeros(ComplexF64, nband, nband, nsymmetry, nkirr)
    for ik in 1:nkirr
        for isym in 1:nsymmetry
            for j in 1:nwannier, i in 1:nwannier
                d_matrix_wann[i, j, isym, ik] = Complex(parse.(Float64, strip.(split(readline(file), ","), Ref(['(', ')', '\n', ' '])))...)
            end
            readline(file)
        end
    end
    for ik in 1:nkirr
        for isym in 1:nsymmetry
            for j in 1:nband, i in 1:nband
                d_matrix_band[i, j, isym, ik] = Complex(parse.(Float64, strip.(split(readline(file), ","), Ref(['(', ')', '\n', ' '])))...)
            end
            readline(file)
        end
    end
    close(file)
    (; nsymmetry, nkirr, ik_to_ikirr, ikirr_to_ik, ikirr_isym_to_isk, d_matrix_wann, d_matrix_band)
end

"""
    function parse_single_value(win_text, keyword, ::Type{T}, default=nothing::Union{Nothing, T}) where {T}
Parse a single value of type `T` from a line "keyword = value" or "keyword : value".
If default is not set and keyword is not found, throw error.
"""
function parse_single_value(win_text, keyword, ::Type{T}, default=nothing::Union{Nothing, T}) where {T}
    i = findfirst(line -> occursin(keyword, line), win_text)
    if i === nothing
        # keyword not found in win_text
        if default === nothing
            error("Required keyword $keyword not found")
        else
            default
        end
    else
        # keyword found in win_text
        if T === String
            split(win_text[i], (':', '='))[end]
        else
            parse(T, split(win_text[i], (':', '='))[end])
        end
    end
end

"""
    read_win(seedname)
Parse `seedname.win` file.
"""
function read_win(seedname)
    println("Reading $seedname.win")
    win_text = strip.(split(read("$seedname.win", String), "\n"))
    filter!(line -> length(line) != 0 && line[1] != '!' && line[1] != '#', win_text)

    # Parse num_wann and num_bands
    nwannier = parse_single_value(win_text, "num_wann", Int)
    nband = parse_single_value(win_text, "num_bands", Int, nwannier)

    # Parse mp_grid
    mp_grid_text = parse_single_value(win_text, "mp_grid", String)
    ngrid = parse.(Int, split(mp_grid_text))
    nktot = prod(ngrid)

    # Parse unit_cell_cart
    i = findfirst(line -> occursin("unit_cell_cart", line), win_text)
    if i === nothing
        error("unit_cell_cart block not found")
    else
        line = lowercase(win_text[i+1])
        if occursin("ang", line) || occursin("bohr", line)
            unit = line
            i += 1
        end
        a1 = parse.(Float64, split(win_text[i+1]))
        a2 = parse.(Float64, split(win_text[i+2]))
        a3 = parse.(Float64, split(win_text[i+3]))
        lattice = Mat3([a1 a2 a3])
        # If unit is Bohr, convert to Angstrom
        if unit == "bohr"
            lattice = ustrip.(auconvert.(u"Å", lattice))
        end
    end

    # Parse kpoints
    kpts = Vec3{Float64}[]
    i = findfirst(line -> occursin("kpoints", line), win_text)
    if i === nothing
        error("kpoints block not found")
    else
        for line in win_text[i+1:i+nktot]
            xk = parse.(Float64, split(line))
            push!(kpts, Vec3(xk))
        end
    end

    (; nwannier, nband, ngrid, nktot, kpts, lattice)
end
