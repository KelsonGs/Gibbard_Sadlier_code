include("./HB_Est.jl")
include("./SimData.jl")
include("./Utils.jl")

using .HB, .SimData, .Utils
using LinearAlgebra, Dates, StatsBase


const n_alt_quest = 2
const n_attribute = 10 # 
const n_subject = 150 # 150
const max_q = 30
const n_question = 30 # 30

const n_holdout = 1000
const n_alternative = 2^n_attribute
const params = [[1,1],[1,2],[2,1],[2,2]]
const config_str = ["LALH","LAHH","HALH","HAHH"]

const p_rand = 0.20

const case = ""
const ref_name = "hybrid_q1"
const path = "<path to directory>"

const rand_questions = [1] 

const mode = "run" # Set to generate for data generation, set to run to run sims. Ensure case is set correctly. 

const mu_prior = 0
const sig_prior = 10

# Note above prior only applied to question selection.
# Will need to manually change HB est. Currently: 0,5 mu, 0,5 sig. 

### Generate Data.
function generate_data(exp_id_path)

    data_mat_0, alt_mat, choice_mat_0 = SimData.gen_data_mat(n_attribute)
    
    
    for param in params
        
        seed = rand(1:1000)

        acc = param[1]
        het = param[2]
        save_path = gen_path_string(exp_id_path,param,"_data")
        
        for sim in 1:n_sim
            seed_val = seed*sim
            data_mat_hold, holdout_data, choice_mat_hold = SimData.data_mat_holdout(data_mat_0,choice_mat_0,n_holdout)
            X_data, Y_data, beta_true_mat, error_mat = SimData.gen_simulation2(data_mat_hold, choice_mat_hold, alt_mat, n_attribute, n_subject, n_question, acc, 
            het,seed_val,mu_prior,sig_prior,case)
            
            save_sim_data(X_data,Y_data,beta_true_mat,error_mat,holdout_data,save_path,sim)
        end
    end
end

function gen_data_array(data_path,seed,config)
    println("Preprocessing 1...")
    flush(stdout)
    data_mat_0, alt_mat, choice_mat_0 = SimData.gen_data_mat(n_attribute)
    
    sim = config[2]
    setting = config[1]
    param = params[setting]
    
    acc = param[1]
    het = param[2]
    
    save_path = gen_path_string(data_path,param,"_data")
    println("Preprocessing 2...")
    flush(stdout)
    data_mat_hold, holdout_data, choice_mat_hold = SimData.data_mat_holdout(data_mat_0,choice_mat_0,n_holdout)
    println("Generating data...")
    flush(stdout)
    X_data, Y_data, beta_true_mat, error_mat = SimData.gen_simulation2(data_mat_hold, choice_mat_hold, alt_mat, n_attribute, n_subject, n_question, acc,het,seed,mu_prior,sig_prior,case,rand_questions)
    save_sim_data(X_data,Y_data,beta_true_mat,error_mat,holdout_data,save_path,sim)
    println("Saving...")
    flush(stdout)
end

            
#run_sims(@__DIR__)
### Run Simulations
function run_sims(exp_id_path,temp_sim_path,config)
    #flush(STDOUT)
    #println("Acc = "*string(param[1]))
    #println("Het = "*string(param[2]))
    #flush(Base.stdout)
    sim = config[2]
    setting = config[1]

    save_path = gen_path_string(exp_id_path,params[setting],"_data")

    seed = rand(1:50000)

    save_path_out = save_path*"/sim"*string(sim)*"/"
    X_data, Y_data, holdout_data, beta_true, error_mat = read_sim_data(save_path_out,"data.h5")
    XY_data = SimData.partition_data(X_data,Y_data,n_attribute,n_subject,n_question,max_q)
    

    stan_path = gen_path_string(temp_sim_path,params[setting],"_sims")
    out_dir_path = stan_path*"/sim$sim"
    isdir(out_dir_path) && rm(out_dir_path; recursive=true)
    if !isdir(out_dir_path)
        mkdir(out_dir_path)
    end
    summary, samples, gen_quants = HBstan_sampler_est_sigma_single(data=XY_data,sim=sim,n_subject=n_subject,
    n_question=n_question,
    n_attribute=n_attribute,
    verbose=true,out_dir=out_dir_path,seed=seed)
    
    
    
    data_outpath = joinpath(temp_sim_path,ref_name)
    if !isdir(data_outpath)
        try
            mkdir(data_outpath)
        catch
            print("Data outpath exists.")
        end
    end
    
    save_sim_results(data_outpath,summary,samples,gen_quants,sim,config_str[setting])
    print(summary)
end


function gen_rand_data(exp_id_path)
    n_sim=50
    
    data_mat_0, alt_mat, choice_mat_0 = SimData.gen_data_mat(n_attribute)
    
    #n_choice = size(data_mat_0,1)
    
    for param in params
        
        seed = sample(1:10000)
        
        
        
        acc = param[1]
        het = param[2]
        save_path = gen_path_string(exp_id_path,param,"_data")
        
        for sim in 1:n_sim
            seed_val = seed*sim
            data_mat_hold, holdout_data, choice_mat_hold = SimData.data_mat_holdout(data_mat_0,choice_mat_0,n_holdout)
            n_choice = size(data_mat_hold,1)
            
            X_data, Y_data, beta_true_mat, error_mat = SimData.gen_randsim(data_mat_hold, choice_mat_hold, alt_mat, n_attribute, n_subject, n_question, acc, het,seed_val,n_choice,case)
            
            save_sim_data(X_data,Y_data,beta_true_mat,error_mat,holdout_data,save_path,sim)
        end
    end
end
#time_id = string("expID_"*Dates.format(now(),"dd-HH-MM"))
#out_dir = mkpath(path*time_id*"/")

# generate_data(out_dir)
function main_sims_exec()
    data_dir = path*"/$ref_name"*"/"
    temp_sim_dir = "/nesi/nobackup/uoo03785/$ref_name"*"/"
    if !isdir(temp_sim_dir)
        try
            mkdir(temp_sim_dir)
        catch
            print("Sim dir already exists.")
        end
    end
    job_id = parse(Int64, ARGS[1])
    sims_config = parse(Int64, ARGS[2])
    #sim_num=1
    #println(ARGS[1])
    config = arrayjob_config(job_id,sims_config)
    
    run_sims(data_dir,temp_sim_dir,config)
end
    
function main_data_exec()
    data_dir = path*"/$ref_name"*"/"
    if !isdir(data_dir)
        try
            mkdir(data_dir)
        catch
            print("File already exists.")
        end
    end
    job_id = parse(Int64, ARGS[1])
    sims_config = parse(Int64, ARGS[2])
    #sim_num=1
    #println(ARGS[1])
    config = arrayjob_config(job_id,sims_config)
    
    gen_data_array(data_dir,job_id,config)
end


function arrayjob_config(args,sims_config)
    args_1 = args-1
    config = divrem(args_1,sims_config).+1
    return config
end

#main_data_exec()

#data_dir = path*"/$ref_name"*"/"
#if !isdir(data_dir)
#    mkdir(data_dir)
#end
#gen_rand_data(data_dir)

if mode == "generate"
    main_data_exec()
elseif mode == "run"
    main_sims_exec()
else
    print("Set 'mode' parameter to one of: 'generate' , 'run'")
end

