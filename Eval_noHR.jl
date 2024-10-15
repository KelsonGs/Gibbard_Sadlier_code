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


const ref_name = "hybrid_q1_2_3_4_5"
const scalr = 1

const res_path = "/<path_to_results>/$ref_name"*"/$ref_name"*"/" # Adjust these according to your pc
const tmp_data_path = "/<path_to_remp_data>/$ref_name"*"/"       # Adjust these according to your pc
const met_save_path = "/<path_to_metrics>/res_$ref_name"*"/"     # Adjust these according to your pc



if !isdir(met_save_path)
    try
        mkdir(met_save_path)
    catch
        println("The path exists.")
    end
end


function filter_files(path::String,type::String,config::String)
    key = Regex("$type+.*$config+.*")
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
    end
    if merge
        return vcat(file_data...)
    else
        return file_data
    end
end

function sort_files(file_list)
    sorted = []
    indexes = []
    for i in 1:sim_lim
        key = Regex("_$i[.]csv")
        for file in file_list
            if occursin(key,file)
                push!(sorted,file)
                push!(indexes,i)
            end
        end
    end
    return sorted,indexes
end


function main_exec(config)
    
    s_files = filter_files(res_path,"sum",config)
    q_files = filter_files(res_path,"quant",config)
    
    sum_files, ind_s = sort_files(s_files)
    quant_files, ind_q = sort_files(q_files)
    println("Files: "*string(length(ind_s)))
    flush(stdout)
    if ind_s != ind_q
        return "Error! - Some sims are missing result files"
    end
        
    summary_pre = read_file(res_path,sum_files;merge=false)
    #summary = [fix_summary(df) for df in summary_pre]
    println("Summary: Done")
    flush(stdout)
    
    quants = read_file(res_path,quant_files;merge=false)
    println("Quantities: Done")
    flush(stdout)
    
    data = [read_sim_data(data_path,"sim$i/data.h5") for i in ind_s]
    println("Data: Done")
    flush(stdout)
    
    n_sim = length(ind_q)
    
    
    hold = [data[i][3] for i in 1:n_sim]
    betas = [data[i][4] for i in 1:n_sim]
    
    
    df = gen_metrics(holdout_data=hold,
    summary_sims = summary_pre,
    beta_true=betas,
    gen_quant=quants,
    config=config,
    save=true,
    n_sim=n_sim)

    print(df)
    flush(stdout)

end

function gen_metrics(;holdout_data,summary_sims, beta_true,gen_quant,config,save=true,n_sim)
    
    rmse = zeros(n_sim)
    ms_sv = zeros(n_sim)
    hr_sv = zeros(n_sim)
    
    ind = findall(x->x == config,config_str)[1]
    params = config_num[ind]
    acc = params[1]
    het = params[2]    

    start_index=8

    #adj_hr_indiv, allSim_hits = adjusted_hit_rate(sample_sims=sample_sims,n_attribute=n_attribute,n_subject=n_subject,n_holdout=n_holdout,holdout_data=holdout_data,
    #beta_true=beta_true,n_sim=n_sim)
    #allSim_hr = mean([mean(allSim_hits[i,:]) for i in 1:n_subject*n_sim])
    
    println("RMSE vals")
    flush(stdout)

    allSim_nbeta, bias_nbeta = normed_RMSE_beta_sims(summary_sims,beta_true,n_attribute,n_subject,n_sim)

    rmse_mu, allSim_rmse_mu, bias_mu, sd_mu = RMSE_mu(acc=acc,n_subject=n_subject,n_attribute=n_attribute,sum_mat=summary_sims,n_sim=n_sim)

    rmse_sig, allSim_rmse_sig, bias_sig, sd_sig = RMSE_Sigma(het=het,acc=acc,n_subject=n_subject,n_attribute=n_attribute,sum_mat=summary_sims,n_sim=n_sim)
    
    rmse_beta, allSim_rmse_beta, bias_beta, sd_beta = calculate_RMSE_beta(summary_sims,beta_true,n_attribute,n_subject,n_sim)
    
    
        #println("Marketshare")
    flush(stdout)
    rmse_indv, errors = MarketShare_Population_RMSE(acc=acc,het=het,holdout_data=holdout_data,n_attribute=n_attribute,n_holdout=n_holdout,gen_quant_sims=gen_quant,n_sim=n_sim,scalr=scalr)
    allSim_MAE = mean(abs.(errors))
    allSim_RMSE = sqrt(mean((errors).^2))
    #println("Hit Rate")
    flush(stdout)
    
    for sim in 1:n_sim
        rmse[sim] = normed_RMSE_beta(summary=summary_sims[sim],beta_true=beta_true[sim],n_attribute=n_attribute,n_subject=n_subject)

        sum_mat = summary_sims[sim]
        β_hat = sum_mat[start_index:start_index+n_attribute*n_subject-1,2]
        β_hat = reshape(β_hat, n_subject,n_attribute)
        
        trueb = [beta_true[sim][i,:] for i in 1:n_subject]
        estb = [β_hat[i,:] for i in 1:n_subject]
        
        trueb = [beta_true[sim][i,:] for i in 1:n_subject]
        estb = [β_hat[i,:] for i in 1:n_subject]
        
        questions = [holdout_data[sim][i,:] for i in 1:n_holdout]
        
        ms_sv[sim] = mean(marketshare(trueb,estb,questions))
        hr_sv[sim] = mean([hitrate(trueb[i],estb[i],questions) for i in 1:n_subject])
    end

    df = DataFrame(hr_sv=hr_sv,ms_sv=ms_sv,rmse_MS=rmse_indv,
    rmse_beta=rmse_beta,
    rmse_mu=rmse_mu,
    rmse_sig=rmse_sig,
    MAE_MS=allSim_MAE,
    RMSE_MS=allSim_RMSE,
    RMSE_nbeta=allSim_nbeta,
    bias_nbeta = bias_nbeta,
    RMSE_beta = allSim_rmse_beta,
    bias_beta = bias_beta,
    sd_beta = sd_beta,
    RMSE_Sigma=allSim_rmse_sig,
    bias_sig=bias_sig,
    sd_sig = sd_sig,
    RMSE_mu=allSim_rmse_mu,
    bias_mu=bias_mu,
    sd_mu = sd_mu)
    println("Saving")
    flush(stdout)
    if save
        info_sheet = DataFrame(acc=acc,het=het,n_attribute=n_attribute,n_subject=n_subject,n_question=n_question,n_holdout=n_holdout)
        save_xlsx(name_str="metrics",file=df,acc=acc,het=het,n_question=n_question,info_sheet=info_sheet,out_dir=met_save_path)
    end
    
    return df
end

#Written for 1 sim so sum mat must be summary_sims[sim]
function calculate_RMSE_beta(sum_mat,beta_true_df,n_attribute,n_subject,n_sim)
    start_index = 8
    errors = Vector{Float64}();
    rmse_sim = Vector{Float64}();
    sds = Vector{Float64}();
    bias_all = Vector{Float64}();
    for i in 1:n_sim

        summary = sum_mat[i]
        # Retrieve mean of betas for Beta[attribute,subject]
        β_hat = summary[start_index:start_index+n_attribute*n_subject-1,2]
    
        # Reshape s.t. we have the same form as our β_true (beta_true_df) matrix
        β_hat = reshape(β_hat, n_subject,n_attribute)
        # Convert to matrix as DataFrames are restricted in using row/col-wise operations like sum and abs
        β_true = Matrix(beta_true_df[i])
        
        error = β_hat .- β_true
        
        sqr_error = error.^2
        vec_sq_err = reshape(sqr_error,n_attribute*n_subject,1)
        
        RMSE_sim = sqrt(mean(vec_sq_err))
        
        sd = summary[start_index:start_index+n_attribute*n_subject-1,4]
        
        append!(sds,sd)
        append!(errors,vec_sq_err)
        append!(rmse_sim,RMSE_sim)
        append!(bias_all,error)
    
    end
    
    rmse_beta = sqrt(mean(errors))
    sd_beta = mean(sds)
    bias_beta = mean(bias_all);
    

    
    return rmse_sim, rmse_beta, bias_beta, sd_beta
end

        
#Written for 1 sim so sum mat must be summary_sims[sim]
function RMSE_mu(;acc,n_subject,n_attribute,sum_mat,n_sim)
    start_index = n_subject*n_attribute+8
    indiv_sim = Vector{Float64}();
    all_sim = Vector{Float64}();
    sd_allsim = Vector{Float64}();
    for i in 1:n_sim
        summary = sum_mat[i]
        # Retrieve mean of betas for Beta[attribute,subject]
        μ_hat_error = (summary[start_index:start_index+n_attribute-1,2] .- μ[acc])
        sd_mu = summary[start_index:start_index+n_attribute-1,4]
        RMSE_mu_val = sqrt(mean((μ_hat_error).^2))
        
        append!(indiv_sim,RMSE_mu_val)
        append!(sd_allsim,sd_mu)
        append!(all_sim,μ_hat_error)
    end
    
    all_sim_RMSE = sqrt(mean((all_sim).^2))
    all_sim_bias = mean(all_sim)
    sds = mean(sd_allsim)
    
    return indiv_sim, all_sim_RMSE, all_sim_bias, sds
end

#Written for 1 sim so sum mat must be summary_sims[sim]
function RMSE_Sigma(;het,acc,n_subject,n_attribute,sum_mat,n_sim)
    
    start_index = n_subject*n_attribute+8+n_attribute
    Sigma_err_sims = Vector{Float64}();
    RMSE_sig_indiv = Vector{Float64}();
    sd_sig = Vector{Float64}();
    true_sig = sqrt(Σ[het]*μ[acc])
    
    for i in 1:n_sim
        summary = sum_mat[i]
        error = (summary[start_index:start_index+n_attribute-1,2] .- (true_sig))
        sd = summary[start_index:start_index+n_attribute-1,4]
        Sigma_err_sims = vcat(Sigma_err_sims,error)
        append!(RMSE_sig_indiv,sqrt(mean(error.^2)))
        append!(sd_sig,sd)
    end
    bias_all_sim = mean(Sigma_err_sims)
    RMSE_all_sim = sqrt(mean((Sigma_err_sims).^2))
    sds = mean(sd_sig)
    
    return RMSE_sig_indiv, RMSE_all_sim, bias_all_sim, sds
end

# Calculates SV rmse
function normed_RMSE_beta(;summary,beta_true,n_attribute,n_subject)
    start_index = 8
    # Retrieve mean of betas for Beta[attribute,subject]
    β_hat = summary[start_index:start_index+n_attribute*n_subject-1,2]
    # Reshape s.t. we have the same form as our β_true (beta_true_df) matrix
    β_hat = reshape(β_hat, n_subject,n_attribute)
    # Convert to matrix as DataFrames are restricted in using row/col-wise operations like sum and abs
    β_true = Matrix(beta_true)
    #Abs value sums for each individual in our sample
    sum_abs_beta_est_vec = sum(abs.(β_hat),dims=2)
    sum_abs_beta_true_vec = sum(abs.(β_true),dims=2)
    #normalised values for the two matrices
    norm_beta_est_sv_mat = n_attribute*β_hat./sum_abs_beta_est_vec
    norm_beta_true_sv_mat = n_attribute*β_true./sum_abs_beta_true_vec
    #reshape the two normalised matrices into vectors
    norm_beta_est_sv_vec =  reshape(norm_beta_est_sv_mat,n_attribute*n_subject,1)
    norm_beta_true_sv_vec = reshape(norm_beta_true_sv_mat,n_attribute*n_subject,1)
    # Calculate normalised RMSE
    RMSE_beta = sqrt(mean((norm_beta_est_sv_vec-norm_beta_true_sv_vec).^2))
    return RMSE_beta
end

function normed_RMSE_beta_sims(summary_sims, beta_true_sims,n_attribute,n_subject,n_sim)
    start_index = 8
    estimated_betas = Matrix{Float64}(undef,0,n_attribute)
    true_betas = Matrix{Float64}(undef,0,n_attribute)
    for i in 1:n_sim
        summary = summary_sims[i]
        # Retrieve mean of betas for Beta[attribute,subject]
        β_hat = summary[start_index:start_index+n_attribute*n_subject-1,2]
        # Reshape s.t. we have the same form as our β_true (beta_true_df) matrix
        β_hat = reshape(β_hat, n_subject,n_attribute)
        # Convert to matrix as DataFrames are restricted in using row/col-wise operations like sum and abs
        β_true = Matrix(beta_true_sims[i])

        estimated_betas = vcat(estimated_betas,β_hat)
        true_betas = vcat(true_betas,β_true)
    end

    #Abs value sums for each individual in our sample
    sum_abs_beta_est_vec = sum(abs.(estimated_betas),dims=2)
    sum_abs_beta_true_vec = sum(abs.(true_betas),dims=2)
    #normalised values for the two matrices
    norm_beta_est_sv_mat = (n_attribute)*estimated_betas./sum_abs_beta_est_vec
    norm_beta_true_sv_mat = (n_attribute)*true_betas./sum_abs_beta_true_vec
    #reshape the two normalised matrices into vectors
    norm_beta_est_sv_vec =  reshape(norm_beta_est_sv_mat,n_attribute*n_subject*n_sim,1)
    norm_beta_true_sv_vec = reshape(norm_beta_true_sv_mat,n_attribute*n_subject*n_sim,1)
    # Calculate normalised RMSE
    RMSE_beta = sqrt(mean((norm_beta_est_sv_vec-norm_beta_true_sv_vec).^2))
    bias_beta = mean(norm_beta_est_sv_vec.-norm_beta_true_sv_vec)
    return RMSE_beta, bias_beta
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
    # Return indiv sim hr and hit rate across all sims for n_holdout*n_sim. 
    return hit_rate_vec_sim, all_hits
end

function MarketShare_Population_Errors(;acc,het,holdout_data,n_attribute,n_holdout,gen_quant_sims,n_sim,scalar)
    
    errors= Vector{Float64}();

    rmse_indiv_sims = zeros(n_sim)
    
    μ_true = μ[acc]*ones(n_attribute)
    Σ_true = Matrix(Σ[het]*μ[acc]*I,n_attribute,n_attribute)
    
    Logit_d = Logistic(0,1)
    
    MvN_D_L = MvNormal(μ_true[1:5], Σ_true[1:5,1:5])
    MvN_D_R = MvNormal(μ_true[1:5]*scalar, Σ_true[1:5,1:5])
    
    println("Marketshares")
    flush(stdout)
    @views @simd for sim in 1:n_sim
        
        gen_β = gen_quant_sims[sim]
        
        
        ϵ = rand(Logit_d,(n_draw,n_holdout))
        # True MS
        β_t_L = transpose(rand(MvN_D_L, n_draw))
        β_t_R = transpose(rand(MvN_D_R, n_draw))
        β_t = hcat(β_t_L,β_t_R)
        
        
        M_t = (β_t*transpose(holdout_data[sim])) .>= 0
        
        
        true_MS = mean.(eachcol(M_t))
        
        # Logistic function: x is β(x-y)
        
        β_p = Matrix(gen_β)
        
        cond_prob = p.(β_p*transpose(holdout_data[sim]))
        
        pred_MS = mean.(eachcol(cond_prob))
        
      
        
        RMSE_sim = sqrt(mean((true_MS - pred_MS).^2))

        err = true_MS-pred_MS

        rmse_indiv_sims[sim] = RMSE_sim
        errors = vcat(errors,err)
        println("$sim / $n_sim")
        flush(stdout)
    end
    # Return RMSE MS
    return rmse_indiv_sims, errors
    #return sqrt(mean((true_MS - pred_MS).^2))
end

function MarketShare_Population_RMSE(;acc,het,holdout_data,n_attribute,n_holdout,gen_quant_sims,n_sim,scalr)
    
    errors= Vector{Float64}();

    rmse_indiv_sims = zeros(n_sim)
    
    μ_true = μ[acc]*ones(n_attribute)
    Σ_true = Matrix(Σ[het]*μ[acc]*I,n_attribute,n_attribute)
    
    
    MvN_D_L = MvNormal(μ_true[1:5], Σ_true[1:5,1:5])
    MvN_D_R = MvNormal(μ_true[1:5]*scalr, Σ_true[1:5,1:5])
    
    Logit_d = Logistic(0,1)
    
    println("Marketshares")
    flush(stdout)
    for sim in 1:n_sim
        
        gen_β = gen_quant_sims[sim]
        
        
        ϵ = rand(Logit_d,(n_draw,n_holdout))
        # True MS
        β_t_L = transpose(rand(MvN_D_L, n_draw))
        β_t_R = transpose(rand(MvN_D_R, n_draw))
        β_t = hcat(β_t_L,β_t_R)
        
        M_t = (β_t*transpose(holdout_data[sim])) +ϵ .>= 0
        
        true_MS = mean.(eachcol(M_t))
        
        # Logistic function: x is β(x-y)
        
        β_p = Matrix(gen_β)
        
        cond_prob = p.(β_p*transpose(holdout_data[sim]))
        
        pred_MS = mean.(eachcol(cond_prob))
        
      
        
        RMSE_sim = sqrt(mean((true_MS - pred_MS).^2))

        err = true_MS-pred_MS

        rmse_indiv_sims[sim] = RMSE_sim
        errors = vcat(errors,err)
        println("$sim / $n_sim")
        flush(stdout)
    end
    # Return RMSE MS
    return rmse_indiv_sims, errors
    #return sqrt(mean((true_MS - pred_MS).^2))
end

function marketshare(truebetas, estimatedbetas, questions)
    abserrors = Float64[]
    for q in questions
        trueshare = 0
        for b in truebetas
            if dot(b,q) >=0
                trueshare+=1
            end
        end
        trueshare /= length(truebetas)
        estimatedshare = 0
        for b in estimatedbetas
            if dot(vec(b),q) >=0
                estimatedshare+=1
            end
        end
        estimatedshare /= length(estimatedbetas)
        push!(abserrors,abs(estimatedshare-trueshare))
    end
    return abserrors
end

function hitrate(truebeta, estimatedbeta, questions)
    hits=0
    for q in questions
        if dot(truebeta,q)*dot(vec(estimatedbeta),q) >=0
            hits+=1
        end
    end
    return hits/length(questions)
end

function save_xlsx(;name_str,file,acc,het,n_question,info_sheet,out_dir)
    acc_str = ""
    het_str = ""
    
    if acc == 2
        acc_str = "HA"
    else
        acc_str = "LA"
    end
    if het == 2
        het_str = "HH"
    else
        het_str = "LH"
    end
    het_acc_sim = acc_str*het_str
    filestr = string(out_dir)*"/"*name_str*"_"*het_acc_sim*"_"*string(n_question)*".csv"
    infostr = string(out_dir)*"/"*"Info_$het_acc_sim"*".csv"
    CSV.write(filestr,file)
    CSV.write(infostr,info_sheet)
    # XLSX.openxlsx(string(out_dir)*"/"*name_str*"_"*het_acc_sim*"_"*string(n_question)*".xlsx",mode="w") do xf
    #     sh = xf["Sheet1"]
    #     XLSX.addsheet!(xf, "Info2")
    #     sh2 = xf["Info2"]
    #     XLSX.writetable!(sh, file)
    #     XLSX.writetable!(sh2, info_sheet)
    # end
end
    



function fix_summary(df)
    if size(df,2) <= 10
        return df
    end
    println("Cleaning summary...")
    flush(stdout)
    for i in 8:1507
        col = df[!,"parameters"][i]
        col_name = col*","*df[!,2][i]
        df[i,1] = col_name
    end

    newdf = empty(df)[:,1:10]
    resize!(newdf,1537)
    newdf.mean .= "0"
    newdf.mean = parse.(Float64,newdf.mean)

    newdf.parameters = df[:,1]

    shifted_vals = Matrix(df[8:1507,3:11])
    top_vals = df[1:7,2:10]
    bot_vals = df[1508:1537,2:10]
    top_vals.mean = parse.(Float64,top_vals.mean)
    bot_vals.mean = parse.(Float64,bot_vals.mean)
    newdf[8:1507,2:10] = shifted_vals
    newdf[1:7,2:10] = top_vals
    newdf[1508:1537,2:10] = bot_vals
    return newdf
end


index = parse(Int64,ARGS[1])
config = config_str[index]

data_str = config*"_data/"
const data_path = joinpath(tmp_data_path,data_str)

if isdir(res_path) && isdir(data_path) && isdir(met_save_path)
    println("Starting...")
    flush(stdout)
    main_exec(config)
else
    println("Failed. No such path(s).")
    println(res_path)
    println(data_path)
    println(met_save_path)
    flush(stdout)
end



# TO RUN:
#RMSE mu and sigma - SV run with 12 Att
#
# Decompose Pop RMSE into bias error. 
#1 10 Sim ellip 30q 10at but with 5k draw 1k warmup
# Above 1.1 Rhat warmup. Above 1.01 is slightly problematic. 
# Gelman and Vitari on Diagnositics - Gives details on n_eff 
# qoute page 5 of http://www.stat.columbia.edu/~gelman/research/published/rhat.pdf

# 1K Warmup 5k Draw - Do 16Q then 30Q, There will be some improvement but should still fail in 
# both cases for n_eff and r_hat but provides some motivation for runs. 
