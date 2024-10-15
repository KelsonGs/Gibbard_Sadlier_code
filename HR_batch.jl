include("./Utils.jl")

using Distributions, DataFrames, LinearAlgebra, CSV, Base.Threads, ProgressBars
using .Utils

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


const sim_class = "hybrid_q1"

const tmp_data_path = "<path_to_temp_data>/$sim_class"*"/$config"*"_data/" # This will need to be changed
const res_path = "/<res_path>/$sim_class"*"/$sim_class"*"/"                # This will need to be changed

if !isdir(res_path)
    print("No such res path.")
    Base.exit()
end



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

function main_exec(config,index_id,res_path)
    
    println(res_path)
    flush(stdout)
    
    s_files = filter_files(res_path,"sum",config)
    q_files = filter_files(res_path,"quant",config)
    samp_files = filter_files(res_path,"samples",config)
    
    sum_files, ind_s = sort_files(s_files,index_id)
    quant_files, ind_q = sort_files(q_files,index_id)
    sample_file, ind_samp = sort_files(samp_files,index_id)
    
    if length(ind_s) != 1
        return "No such sim ID: $index_id"
    end
        
    
    println("Length samples: "*string(length(sample_file)))
    flush(stdout)
    
    
    samples = read_file(res_path,sample_file;merge=false)
    println("Samples: Done")
    flush(stdout)
    
    
    data = [read_sim_data(tmp_data_path,"sim$i/data.h5") for i in ind_s]
    println("Data: Done")
    flush(stdout)
    
    n_sim = 1
    
    hold = [data[i][3] for i in 1:n_sim]
    betas = [data[i][4] for i in 1:n_sim]
    
    hits = adjusted_hit_rate(sample_sims = samples,n_attribute=n_attribute,n_subject=n_subject,n_holdout=n_holdout,
    holdout_data=hold,beta_true=betas,n_sim=n_sim)
    
    res_path = "./ResultsHR/$sim_class"
    
    if !isdir(res_path)
        mkdir(res_path)
    end
    
    res_path_full = "$res_path"*"/$config"*"/"
    
    if !isdir(res_path_full)
        mkdir(res_path_full)
    end
    
    save_hr_data(res_path_full,hits,index_id)

end

function arrayjob_config(args,sims_config)
    args_1 = args-1
    config = divrem(args_1,sims_config).+1
    return config
end


p(x) = 1/(1+exp(-x))
# ====================== Adjusted Marketshare Calculation ========================
#Based on Peters notes - see Design.pdf
function adjusted_hit_rate(;sample_sims,n_attribute,n_subject,n_holdout,holdout_data,beta_true,n_sim)
    
    hit_rate_vec_sim = zeros(n_sim)
    all_hits = BitArray(undef,0,n_holdout)
    
    println("Hit Rate")
    flush(stdout)
    scalar = 1/n_draw
    
    for sim in 1:n_sim

        gen_β = sample_sims[sim]
        sample_beta_vecs = zeros(n_draw,n_attribute,n_subject)
        
        # Need to get each subjects 10x1 beta vector. Each subject has n_draw beta vectors. 
        for a in 1:n_attribute
            for s in 1:n_subject
                str_index = "beta.$a.$s"
                sample_beta_vecs[:,a,s] = @view(gen_β[:,str_index])
            end
        end
        p_y_xy = zeros(n_subject,n_holdout)

        holdout_vec = holdout_data[sim]
        @threads for q ∈ 1:n_holdout
            p_y_xy[:,q] = [sum(scalar .* p.(@view(sample_beta_vecs[:,:,s]) * @view(holdout_vec[q,:]))) for s ∈ 1:n_subject]
        end
        pred_choice = p_y_xy .>= 0.5
        
        true_choice = beta_true[sim]*holdout_data[sim]' .>= 0
        
        hit = true_choice .== pred_choice
        all_hits = vcat(all_hits,hit)
        hit_rate_vec_sim[sim] = mean([mean(hit[i,:]) for i in 1:n_subject])
        println("$sim / $n_sim")
        flush(stdout)
    end
    # Can average across these if desired.

    return all_hits
end




job_id = parse(Int64,ARGS[1])
total_id = parse(Int64,ARGS[2])

setting = arrayjob_config(job_id,total_id)

config = config_str[setting[1]]
sim_number = setting[2]


println(res_path)
flush(stdout)
main_exec(config,sim_number,res_path)
