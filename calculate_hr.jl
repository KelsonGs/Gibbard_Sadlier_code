include("./Utils.jl")

using Distributions, DataFrames, LinearAlgebra, CSV, Base.Threads, ProgressBars
using .Utils
using HDF5

# # ======= Estimation Controls for MCMC (W.O. Σ estimation) ========
const n_chains = 4 #default.  
const N_all = 25000 # 25000 for sigma est
const n_draw = n_chains*N_all # same
const μ = [0.5,1.5]
const Σ = [0.5,2.0]

const config_num = [[1,1],[1,2],[2,1],[2,2]]
const config_str = ["LALH","LAHH","HALH","HAHH"]

const n_alt_quest = 2
const n_attribute = 10 # 
const n_subject = 150 # 150
const n_question = 30 # 30
const n_holdout = 1000
const n_alternative = 2^n_attribute
const sim_lim = 50

const sim_id = "hybrid_q1_7_13_19_25"

function filter_files(path,types,config)
    key = Regex("$types+.*$config+.*")
    file_by_config = filter(x->occursin(key,x), readdir(path))
    return file_by_config
end

function read_file(path,file_names;merge=false)

    file_data = [DataFrame() for i in 1:length(file_names)]
    i = 1
    for file in file_names
        file_path = joinpath(path,file)
        df = CSV.read(file_path,DataFrame,silencewarnings=true,strict=false)
        file_data[i] = df
        i +=1
        println(string(i))
        flush(stdout) 
    end
    
    if merge
        return vcat(file_data...)
    else
        return file_data
    end
    
end


function sort_files(file_list,ind)
    sorted = []
    indexes = []
    key = Regex("_$ind[.]csv")
    for file in file_list
        if occursin(key,file)
            push!(sorted,file)
            push!(indexes,ind)
        end
    end
    return sorted,indexes
end

function main_exec(config)    
    
    all_hits = BitArray(undef,0,n_holdout)
    
    n_sims_found = 0
    for i in 1:50
        dir = res_path*"/sim$i"
      
        if isdir(dir)
            hit = h5read(dir*"/data.h5","hits")
            if length(hit) != 150000
                println("Missing data $i")
                flush(stdout)
            else
                all_hits = vcat(all_hits,hit)
                n_sims_found += 1
            end
        else
            println("Missing sim: $i"*" Res path : $dir")
            flush(stdout)
        end
    end
    
    hitrate = mean([mean(all_hits[i,:]) for i in 1:n_subject*n_sims_found])
    
    println("Experiment ID -- DATA: $sim_id")
    println("-------HR---------")
    println("$hitrate")
    flush(stdout)
    
    #save_hr_data(res_path,hits,index_id)

end

function arrayjob_config(args,sims_config)
    args_1 = args-1
    config = divrem(args_1,sims_config).+1
    return config
end

job_id = parse(Int64,ARGS[1])

config = config_str[job_id]

const res_path = "./ResultsHR/$sim_id"*"/$config"

if !isdir(res_path)
    Base.exit()
end

main_exec(config)
