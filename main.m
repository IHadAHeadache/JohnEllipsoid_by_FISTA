epsilon = 0.0000000000000001; rho = 9999999999;
g = [0 -0.5*3^0.5 0.5*3^0.5; -1 0.5 0.5];
h = [1 1 1];
L = 10;
t = 1/L;
beta = 0.98;
u = [0];
W = [0.5 0;0 0.5]; y = [0; 0];
W(:,:,2) = [W]; y(:,:,2) = [y];
for k = 3:2000000
    u(k-1) = 0.5*(1+(1+4*u(k-2)^2)^0.5);
    temp_W = W(:,:,k-1) + (u(k-2)-1)/u(k-1)*(W(:,:,k-1)-W(:,:,k-2));
    temp_y = y(:,:,k-1) + (u(k-2)-1)/u(k-1)*(y(:,:,k-1)-y(:,:,k-2));
    [Wk,yk] = prox(temp_W,temp_y,epsilon,rho,g,h,t);
    W(:,:,k) = [Wk]; y(:,:,k) = [yk];
    t = 1/L;
end
W(:,:,k-1)
y(:,:,k-1)