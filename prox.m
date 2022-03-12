function [outputW,outputy] = prox(inputW,inputy,epsilon,rho,g,h,t)
    gradient_y = zeros(size(inputy,1),1);
    gradient_W = zeros(size(inputW,1),size(inputW,2));
    for i = 1:size(g,2)
        z = (norm(inputW*g(:,i))^2 + epsilon)^0.5 + transpose(g(:,i))*inputy - h(i);
        if z < 0
            gradient_y = gradient_y + 0*g(:,i);
            gradient_W = gradient_W + 0*inputW*g(:,i)*transpose(g(:,i))/(norm(inputW*g(:,i))^2 + epsilon)^0.5;
        elseif z <= 1
            gradient_y = gradient_y + z*g(:,i);
            gradient_W = gradient_W + z*inputW*g(:,i)*transpose(g(:,i))/(norm(inputW*g(:,i))^2 + epsilon)^0.5;
        else
            gradient_y = gradient_y + 1*g(:,i);
            gradient_W = gradient_W + 1*inputW*g(:,i)*transpose(g(:,i))/(norm(inputW*g(:,i))^2 + epsilon)^0.5;
        end
    end
    outputy = inputy - t*gradient_y;
    W_bar = inputW - t*gradient_W;
    [R,lambda] = eig(((W_bar-epsilon*eye(size(W_bar,2)))+transpose(W_bar-epsilon*eye(size(W_bar,2)))) / 2);
    Diagonal_W = zeros(size(lambda,2));
    for i = 1:size(lambda,1)
        if t+epsilon*rho*lambda(i,i) > 0
            Diagonal_W(i,i) = 0.5*(lambda(i,i)-epsilon+((lambda(i,i)-epsilon)^2+4*(epsilon*lambda(i,i)+t/rho))^0.5);
        end
    end
    outputW = R*Diagonal_W*transpose(R) + epsilon*eye(size(R,1));
end

