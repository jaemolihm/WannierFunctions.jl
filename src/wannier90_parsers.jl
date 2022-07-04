using StaticArrays

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