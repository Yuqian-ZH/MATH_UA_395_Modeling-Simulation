rng(298)
tic
%specify the number of processes and goods
global n m A B r_total r_count;
n = 100;        %process
m = 15;         %good
lambda = 0.3;   %interest rate

%here's the initial variables randomly generated
%%A and B are positive
A = abs(0.1 * randn(n,m) + 10)+1e-5;                  %consumption function
B = abs(0.1 * randn(n,m) + 10)+1e-5;                  %production function
r0 = ones(n,1);                                       %intensity r
q0 = 10 + randn(m,1);                                 
q0 = q0/sqrt(sum(q0.^2));                       

%set up for simulation
t_max = 100;                        %max timestep
q_total = zeros(m,t_max + 1);       %record q
p_total = zeros(m,t_max + 1);       %record p
excess_total = zeros(m,t_max + 1);  %record excess demand
r_total = zeros(n,t_max + 2);       %record intensity
r_total(:,1) = r0;              
r_count = 1;                   
iter_count = zeros(1, t_max + 1);   % count the iteration to get the proper p

fun = @Minimization;            %objective function to minimize

%nonlicon = @nlcon;              %nonlinear constraint
nonlicon = [];
%options for the nonlicon, uncommended for the basic mode
% options = optimoptions('fmincon',"EnableFeasibilityMode",true,'SpecifyObjectiveGradient',true, ...
%     'MaxIterations', 3e3, 'MaxFunctionEvaluations',3e4, "SubproblemAlgorithm","cg",'Display','iter');   
                      
for t = 0:t_max 
    if ((t+1) == 1)
        q_equi = fmincon(fun, q0, [], [], [], [], [], [], nonlicon, options);  %minimize the objective function with equilibiurm price in the previous day
    else
        q_equi = fmincon(fun, q_total(:,t), [], [], [], [], [], [], nonlicon, options);  %minimize the objective function   
    end
    % if fun(q_equi) > 0.01 (which should be zero theoretically, but we have some tolerance in practice),
    % then q_equi is not a global minimizer, then we rerun the minimization
    % with a different starting point

    validResult = false;                         %check
    %minexcessdemand = 100;%initialize current min of excess
    for itr = 1 : 10
        while fun(q_equi) > 0.01
            iter_count(1,t+1)=iter_count(1,t+1)+1;
            %q0 = 10 + 3*randn(m,1);                      
            q0 = 10 + randn(m,1);                   %choose a number for q0
            q0 = q0/sqrt (sum(q0.^2));              
            q_equi = fmincon(fun, q0, [], [], [], [], [], [], nonlicon, options);  %minimize the objective function
            %q_equi = fminsearch(fun,q0);           %with fminsearch method
            disp(ExcessDemand(q_equi.^2));          %dispaly E
            %disp(fun(q_equi));
            %disp('r is');
            %disp(r_total(t+1));
        end
        p_equi = q_equi;                            %equilibrium price
        %record 
        q_total(:,t+1) = q_equi; 
        p_total(:,t+1) = p_equi;
        excess = ExcessDemand(q_equi);
        if(all(excess < .01))                       %check: excess every good < 0
              validResult = true;
              break;
        end
    end


    if(~validResult)                                %report the error if the model cannot work well
          disp("invalid result, there's an error");
          return;
    end

    %r(t+1)
    r_total(:,(t+1)+1) = (r_total(:,t+1) .* ((B*(q_equi.^2)) ./ (A*(q_equi.^2)))) ; 
    % eliminate small variables
    for count = 1:n         %r
        if (r_total(count,(t+1)+1)) < 1e-2 && (r_total(count,t+1)) > 0 || r_total(count,(t+1)+1) < 0
            r_total(count,(t+1)+1) = 0;
        end
    end

    for count = 1:m         %E
        if (excess(count) < 1e-2 && excess(count) > 0) || (excess(count) > -1e-2 && excess(count) < 0)
            excess_total(count,t+1) = 0;
        else
            excess_total(count,t+1) = excess(count);
        end
    end
    r_count = r_count + 1;      %alternative referencing t

    %the choose of con (saving/borrowing) depends on the previous market
    %condition

    if (r_total(:,t+1) .* (B*(q_equi.^2)) > r_total(:,(t+1)+1) .* A*(q_equi.^2))
        %con = repmat(0.002 .* ((B*(q_equi.^2))./(A*(q_equi.^2))),200,1);
        %%another way of chooing c
        con = 0.002;
    else
        %con = repmat(-0.002 .* ((B*(q_equi.^2))./(A*(q_equi.^2))),200,1);
        con = -0.002;
    end
end

toc

%Functions used above:
%excess demand
function [E, grad_E] = ExcessDemand(q)
    global m A B r_total r_count lambda con;
    %excess demand function
    E = ( transpose(r_total(:,r_count)) * ((repmat((lambda * con + B*(q.^2)) ./ (con + A*(q.^2)), 1, m) .* A - B) )).';
    %gradient of excess demand function
    grad_E = 2.*q.*((transpose(r_total(:,r_count)) * ((A.*( ((B.*repmat((con + A*(q.^2)), 1, m))...
        -(A.*repmat((lambda * con + B*(q.^2)), 1, m))) ./ (repmat((con + A*(q.^2)).^2, 1, m)) )))))';
end

%minimize
function [phi, grad] = Minimization(q)
    global m A B r_total r_count lambda con;
    E = ( transpose(r_total(:,r_count)) * ( repmat((B*(q.^2)) ./ (A*(q.^2)), 1, m) .* A - B ) ).'; 
    grad_E = 2.*q.*((transpose(r_total(:,r_count)) * ((A.*( ((B.*repmat((con + A*(q.^2)), 1, m)) ...
        -(A.*repmat((lambda * con + B*(q.^2)), 1, m))) ./ (repmat((con + A*(q.^2)).^2, 1, m)) )))))';

    phi = 0;
    grad = zeros(m,1);
    for e_j = 1:m                      %take the sum of E>0
        if E(e_j) > 0
            phi = phi + E(e_j);
        end
    end

    %gradient descent
    if nargout > 1 
        for k = 1:m
            if (E(k) > 0) 
                grad(k) = grad(k) + (E(k) * (grad_E(k))') ;
            end
        end
    end
end

% for the nonlinear constrain
% uncommend if for the basic model
% function [c,ceq] = nlcon(q)
%     c = [];
%     ceq = sum(q.^2)-1;
% end


