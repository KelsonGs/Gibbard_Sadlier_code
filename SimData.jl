module SimData
export gen_data_mat, data_mat_holdout, gen_simulation2, gen_randsim
using LinearAlgebra
using QuadGK # For numerical integration 
using Combinatorics
using Random, StatsBase
using Distributions
using Interpolations
using BenchmarkTools
using Base.Threads

const μ = [0.5, 1.5]  #first entry low accuracy, second high
const Σ = [0.5, 2]



# Calculation of the g_function
function g_function(d=2,v=1:15,m=-32:32)
    
    #M,V = meshgrid(m,v)
    M = [j for i in eachindex(v), j in m]
    V = [i for i in v, j in eachindex(m)]
    
    n_v = size(V,1)
    n_m = size(M,2)
    
    p_pos = zeros(n_v, n_m)
    E_Z_1_pos = zeros(n_v, n_m)
    E_Z_sqr_1_pos = zeros(n_v, n_m)
    
    p_neg = zeros(n_v, n_m)
    E_Z_1_neg = zeros(n_v, n_m)
    E_Z_sqr_1_neg = zeros(n_v, n_m)
    
    for j = 1: n_v
        for k = 1: n_m
            p_pos[j,k] = quadgk(z -> ((1+ exp(-M[j,k] - V[j,k].*z)).^(-1)).*((2 .*pi).^(-1/2)).*exp(-(z.^2)/2),-Inf,Inf)[1]
            E_Z_1_pos[j,k] = quadgk(z -> z.*((1+ exp(-M[j,k] - V[j,k].*z)).^(-1)).*((2 .*pi).^(-1/2)).*exp(-(z.^2)/2),-Inf,Inf)[1]
            E_Z_sqr_1_pos[j,k] = quadgk(z -> (z.^2).*((1+ exp(-M[j,k] - V[j,k].*z)).^(-1)).*((2 .*pi).^(-1/2)).*exp(-(z .^2)/2),-Inf,Inf)[1]
            p_neg[j,k] = quadgk(z -> ((1+ exp(M[j,k] - V[j,k].*z)).^(-1)).*((2 .*pi).^(-1/2)).*exp(-(z.^2)/2),-Inf,Inf)[1]
            #Relative error tolerance specified here to prevent quadgk from hanging
            E_Z_1_neg[j,k] = quadgk(z -> z.*((1+ exp(M[j,k] - V[j,k].*z)).^(-1)).*((2 .*pi).^(-1/2)).*exp(-(z.^2)/2),-Inf,Inf,rtol=1e-4)[1]
            E_Z_sqr_1_neg[j,k] = quadgk(z -> (z.^2).*((1+ exp(M[j,k] - V[j,k].*z)).^(-1)).*((2 .*pi).^(-1/2)).*exp(-(z .^2)/2),-Inf,Inf)[1]
        end 
    end
    
    Var_pos = E_Z_sqr_1_pos./p_pos - (E_Z_1_pos./p_pos).^2
    Var_neg = E_Z_sqr_1_neg./p_neg - (E_Z_1_neg./p_neg).^2
    
    gf = p_pos.*(Var_pos.^(1 ./d)) + p_neg.*(Var_neg.^(1 ./d))
    return gf
end


# Checked against matlab and ok. 
function gen_data_mat(n_attribute)
    n_alternative = 2^n_attribute

    alternative_mat = [digits(i, base=2, pad=n_attribute) |> reverse for i in 0:n_alternative-1]
    #NOTE: May need to reverse indicies to match matlab output
    alternative_mat_0 = transpose(reduce(hcat,alternative_mat))

    alternative_mat = alternative_mat_0[:, end:-1:1]
    
    choice_combins = combinations(1:n_alternative, 2)

    choice_mat_flat = collect(Iterators.flatten(choice_combins))

    choice_combins_col = collect(choice_combins)
    choice_mat_0 = Int32.(transpose(reshape(collect(Iterators.flatten(choice_combins_col)), (length(choice_combins_col[1]),length(choice_combins_col)))))

    n_choice = div(length(choice_mat_flat), 2) 
    
    
    data_mat = [reduce(vcat, alternative_mat[choice_mat_flat[(i-1)*2+1],:] .- alternative_mat[choice_mat_flat[(i-1)*2+2],:]) for i in 1:n_choice]

    reshape_data = vcat(transpose.(data_mat)...)
    
    data_mat_0 = reshape_data[:, 1:end]
    
    return data_mat_0, alternative_mat, choice_mat_0
end


function data_mat_holdout(data_mat_0,choice_mat_0,n_holdout)

    n_choice_inc_holdout = size(data_mat_0,1)
    holdout_questions_vec = sample(1:n_choice_inc_holdout, n_holdout, replace = false)

    data_holdout = data_mat_0[holdout_questions_vec,:]
    choice_mat = choice_mat_0[setdiff(1:end, holdout_questions_vec), :]
    data_mat = Int8.(data_mat_0[setdiff(1:end, holdout_questions_vec), :])

    return data_mat, data_holdout, choice_mat

end


function gen_Betas(mu_true,sigma_true,seed,n_subject,n_question,n_attribute,case)

    scalar = (1/2) # scale mu_0 
    Random.seed!(seed)

    mu_0 = ones(n_attribute)*mu_true
    Sigma_0 = Matrix(I*sigma_true, n_attribute, n_attribute)
    
    Logit_d = Logistic(0,1)
    error_mat = Float16.(rand(Logit_d,(n_subject,n_question)))
    red_size = divrem(n_attribute,2)
    print(red_size)
    # reduce size of sigma matrix - used in CASE 1 and CASE 2 only.
    # CASE 1: Generate betas for first 5 attributes using given mu_0, generate betas for second 5 attributes by multiplying mu_0 (1/2)
    # - We call this case the "scaled" case as the location paramater is scaled in the second half of the betas. 
    if case == "scaled"

        MvN_d_L = MvNormal(mu_0[1:red_size[1]], Sigma_0[1:red_size[1],1:red_size[1]])
        MvN_d_R = MvNormal(mu_0[1:red_size[1]].*scalar, Sigma_0[1:(red_size[1]+red_size[2]),1:(red_size[1]+red_size[2])])

        beta_true_mat_L = Float16.(rand(MvN_d_L, n_subject)')
        beta_true_mat_R = Float16.(rand(MvN_d_R, n_subject)')
    
        beta_true_mat = hcat(beta_true_mat_L,beta_true_mat_R)

        return beta_true_mat, error_mat
    # CASE 2: Generate betas for first 5 attributes using given mu_0, generate betas for second 5 attributes by multiplying mu_0 by -1. 
    # - We call this case the "signed" case as the location paramater is negatively signed in for part of the betas. 
    elseif case == "signed"
        MvN_d_L = MvNormal(mu_0[1:red_size[1]], Sigma_0[1:red_size[1],1:red_size[1]])
        MvN_d_R = MvNormal(mu_0[1:red_size[1]].*(-1), Sigma_0[1:(red_size[1]+red_size[2]),1:(red_size[1]+red_size[2])])

        beta_true_mat_L = Float16.(rand(MvN_d_L, n_subject)')
        beta_true_mat_R = Float16.(rand(MvN_d_R, n_subject)')
    
        beta_true_mat = hcat(beta_true_mat_L,beta_true_mat_R)

        return beta_true_mat, error_mat
    # CASE 3: This is the default case. All betas are generated using same location parameter.
    else
        MvN_d = MvNormal(mu_0, Sigma_0)

        beta_true_mat = Float16.(rand(MvN_d, n_subject)')
        
        return beta_true_mat, error_mat
    end       
end


function quadgk_update(m_iter,sigma_iter)
    p = quadgk(z -> (1+ exp(-m_iter - sigma_iter*z))^(-1)*exp(-z^2/2)/sqrt(2*pi),-Inf,Inf)[1]
    E_Z = quadgk(z -> z*(1+ exp(-m_iter - sigma_iter*z))^(-1)*exp(-z^2/2)/sqrt(2*pi),-Inf,Inf)[1]/p
    Var_Z = quadgk(z -> z*z*(1+ exp(-m_iter - sigma_iter*z))^(-1)*exp(-z^2/2)/sqrt(2*pi),-Inf,Inf)[1]/p - E_Z^2
    return E_Z, Var_Z
end


function fill_X_arr(n_attribute,n_subject,n_question,n_alt_quest,alt_mat,choice_mat,question_mat)

    X_arr = zeros(n_subject*n_question*n_alt_quest,n_attribute+3)

    for i = 1:n_subject
        for q = 1:n_question
            for a = 1:n_alt_quest
                x_arr_row = [i,q,a]
                index = Int(question_mat[i,q])
                append!(x_arr_row, alt_mat[choice_mat[index,a],:])
                X_arr[(i-1)*n_question*n_alt_quest+(q-1)*n_alt_quest+a,:] = x_arr_row
            end
        end
    end
    return X_arr
end


function gen_sim_data(x_data,y_data, n_attribute, n_subject, n_question)
    # Need to fix this for situation where the cols is reduced due to max_question parameter
    x_sim = zeros(n_subject,n_question,n_attribute) #why is n_attribute dim in here?
    y_sim = zeros(n_subject,n_question)
    
  
    for s in 1:n_subject
      for q in 1:n_question
        y_sim[s,q] = y_data[s,q]
        for a in 1:n_attribute
          x_sim[s,q,a] = x_data[2*n_question*(s-1)+2*q-1, 3+a] -
          x_data[2*n_question*(s-1)+2*q, 3+a]
        end
      end 
    end
  
    y_sim = round.(Int, y_sim)
    return x_sim, y_sim
end



function interpolate_constructor(range_x,range_y,g_mat)
    #
    itp = interpolate((range_x,range_y), g_mat, Gridded(Linear()))

    return itp
end


function get_interp_vals(vec_x,vec_y,sitp)

    estimates= Vector{Float64}(undef,length(vec_x))

    @simd for i in eachindex(vec_x)
        @inbounds estimates[i] = sitp(vec_x[i],vec_y[i])
    end

    min_ind = findmin(estimates)[2]
    return min_ind
end



function gen_simulation2(data_mat, choice_mat, alt_mat, n_att, n_sub, n_q, acc, het,seed,mu_prior,sig_prior,case,rand_questions)
#function gen_simulation2(data_mat, choice_mat, alt_mat, n_att, n_sub, n_q, acc, het,seed,mu_prior,sig_prior,case, p_rand)

    n_alt_quest = 2
    mu_true = μ[acc]
    sigma_true = Σ[het]*μ[acc]

    mu_0 = ones(n_att)*mu_prior
    Sigma_0 = Matrix(I*sig_prior, n_att, n_att)
    
    beta_true_mat, error_mat = gen_Betas(mu_true,sigma_true,seed,n_sub,n_q,n_att,case)
    #print(size(data_mat))
    Question_mat = Matrix{Int32}(undef, n_sub,n_q)
    Y_data = BitArray(undef,n_sub,n_q)

    g_mat = g_function()
    sitp = interpolate_constructor(1:15,-32:32,g_mat)
    
    # Might get lock up from threads calling rng at the same time... 
    rng = MersenneTwister(seed)
    
    @threads for i = 1:n_sub

        #sitp = interpolate_constructor(1:15,-32:32,g_mat)

        @inbounds error_vec = error_mat[i,:]
        @inbounds beta_true_vec = beta_true_mat[i,:]'

        #q,y = find_min_question(Sigma_0,mu_0,data_mat,beta_true_vec,error_vec,n_q,sitp)
        q,y = find_min_question_hybrid(Sigma_0,mu_0,data_mat,beta_true_vec,error_vec,n_q,sitp,rand_questions,n_att)
        #q,y = find_min_question_hybrid_p(Sigma_0,mu_0,data_mat,beta_true_vec,error_vec,n_q,sitp,p_rand,n_att,rng)
        
        Question_mat[i,:] = q
        Y_data[i,:] = y
        #println("Subject: $i")
    end

    X_data = fill_X_arr(n_att,n_sub,n_q,n_alt_quest,alt_mat,choice_mat,Question_mat)

    return X_data, Y_data, beta_true_mat, error_mat
    #return Question_mat, Y_data, holdout_data, beta_true_mat, error_mat
end

function partition_data(X_data, Y_data, n_attribute, n_subject, n_question, max_q)
    x_sim = zeros(n_subject,n_question,n_attribute) #why is n_attribute dim in here?
    y_sim = zeros(n_subject,n_question)
    
  
    for s in 1:n_subject
      for q in 1:n_question
        for a in 1:n_attribute
          x_sim[s,q,a] = X_data[2*max_q*(s-1)+2*q-1, 3+a] -
          X_data[2*max_q*(s-1)+2*q, 3+a]
        end
      end 
    end

    for s in 1:n_subject
        for q in 1:n_question
          y_sim[s,q] = Y_data[s,q]
        end
    end
  
    y_sim = round.(Int, y_sim)
    return x_sim, y_sim
end



function find_min_question(Sigma_0,mu_0,data_mat,beta_true_vec,error_vec,n_question,sitp)

    question_mat = zeros(Int64,n_question)
    y_arr = BitArray(undef,n_question)

    Sigma = Sigma_0
    mu = mu_0

    for q ∈ 1:n_question

        ch_Sigma = cholesky(Hermitian(Sigma)).U
        
        A = data_mat * ch_Sigma

        sigma_vec = [norm(C) for C in eachcol(A')]
    
        m_vec = data_mat*mu
  
        min_ind = get_interp_vals(sigma_vec,m_vec,sitp)


        question_mat[q] = min_ind

        v = data_mat[min_ind,:]

        y_ind = ((beta_true_vec*v) + error_vec[q]) > 0
        y_arr[q] = Int8(y_ind)

        v_iter = ((y_ind*v) -  ((1-y_ind)*v))
        
        m_iter = dot(mu,v_iter)
        sigma_iter = sqrt((v_iter')*(Sigma*v_iter))
 
        E_Z, Var_Z = quadgk_update(m_iter,sigma_iter)


        sv = Sigma*v_iter/sigma_iter
        mu = mu + sv*E_Z
       
        Sigma = Sigma + ((Var_Z .-1)*sv)*sv'


    end
    return question_mat, y_arr
end

function find_min_question_hybrid(Sigma_0,mu_0,data_mat,beta_true_vec,error_vec,n_question,sitp,rand_questions,n_attribute)
    
    question_mat = zeros(Int64,n_question)
    y_arr = BitArray(undef,n_question)

    Sigma = Sigma_0
    mu = mu_0

    for q ∈ 1:n_question
        
        v = zeros(Float64,n_attribute)
        
        if q ∉ rand_questions 

            ch_Sigma = cholesky(Hermitian(Sigma)).U

            A = data_mat * ch_Sigma

            sigma_vec = [norm(C) for C in eachcol(A')]

            m_vec = data_mat*mu

            min_ind = get_interp_vals(sigma_vec,m_vec,sitp)

            question_mat[q] = min_ind #Index the minimum question

            v = data_mat[min_ind,:] # The question itself 
        else
            # Select a random question 
            min_ind = rand(1:length(data_mat[:,1]))

            question_mat[q] = min_ind

            v = data_mat[min_ind,:]
        end
        y_ind = ((beta_true_vec*v) + error_vec[q]) > 0
        y_arr[q] = Int8(y_ind)

        v_iter = ((y_ind*v) -  ((1-y_ind)*v))

        m_iter = dot(mu,v_iter)
        sigma_iter = sqrt((v_iter')*(Sigma*v_iter))

        E_Z, Var_Z = quadgk_update(m_iter,sigma_iter)

        sv = Sigma*v_iter/sigma_iter
        mu = mu + sv*E_Z

        Sigma = Sigma + ((Var_Z .-1)*sv)*sv'

    end
    return question_mat, y_arr
end

function find_min_question_hybrid_p(Sigma_0,mu_0,data_mat,beta_true_vec,error_vec,n_question,sitp,p_rand,n_attribute,rng)
    
    question_mat = zeros(Int64,n_question)
    y_arr = BitArray(undef,n_question)

    Sigma = Sigma_0
    mu = mu_0
    
    for q ∈ 1:n_question
        
        v = zeros(Float64,n_attribute)
        

        random_number = rand(rng)
        
        if random_number > p_rand && q != 1

            ch_Sigma = cholesky(Hermitian(Sigma)).U

            A = data_mat * ch_Sigma

            sigma_vec = [norm(C) for C in eachcol(A')]

            m_vec = data_mat*mu

            min_ind = get_interp_vals(sigma_vec,m_vec,sitp)

            question_mat[q] = min_ind #Index the minimum question

            v = data_mat[min_ind,:] # The question itself 
        else
            # Select a random question 
            min_ind = rand(1:length(data_mat[:,1]))

            question_mat[q] = min_ind

            v = data_mat[min_ind,:]
        end
        y_ind = ((beta_true_vec*v) + error_vec[q]) > 0
        y_arr[q] = Int8(y_ind)

        v_iter = ((y_ind*v) -  ((1-y_ind)*v))

        m_iter = dot(mu,v_iter)
        sigma_iter = sqrt((v_iter')*(Sigma*v_iter))

        E_Z, Var_Z = quadgk_update(m_iter,sigma_iter)

        sv = Sigma*v_iter/sigma_iter
        mu = mu + sv*E_Z

        Sigma = Sigma + ((Var_Z .-1)*sv)*sv'

    end
    return question_mat, y_arr
end

function gen_randsim(data_mat, choice_mat, alt_mat, n_att, n_sub, n_q, acc, het,seed,n_choice,case)
    n_alt_quest = 2
    mu_true = μ[acc]
    sigma_true = Σ[het]*μ[acc]

    
    beta_true_mat, error_mat = gen_Betas(mu_true,sigma_true,seed,n_sub,n_q,n_att,case)
    
    
    
    #print(size(data_mat))
    Question_mat = Matrix{Int32}(undef, n_sub,n_q)
    Y_data = BitArray(undef,n_sub,n_q)

    for i in 1:n_sub
        Question_mat[i,:] = sample(1:n_choice,n_q,replace=false)
    end

    for i in 1:n_sub
        for q in 1:n_q
            x_less_y_row_vec = data_mat[Question_mat[i,q],:]
            y_index = ((beta_true_mat[i,:]'*x_less_y_row_vec) + error_mat[i, q]) > 0
            Y_data[i,q] = y_index
        end
    end
    X_data = fill_X_arr(n_att,n_sub,n_q,n_alt_quest,alt_mat,choice_mat,Question_mat)
    return X_data, Y_data, beta_true_mat, error_mat
end

end



