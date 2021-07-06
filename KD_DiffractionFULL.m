tic
% This code computes the KD diffraction pattern
% following the double slit (full) code
% with a point source.
% Everything in SI units.
clear all
clc
format long

m=9.10938356e-31;
q=1.602e-19;
hbar=1.0545718e-34;
c=299792458;
eps0=8.85e-12;

E=380; % electron energy in eV
v=sqrt(2*(E*q)/m); % electron velocity ~ 1.1e7 m/s
global lamdB
lamdB=2*pi*hbar/(m*v); % deBroglie wavelength

D=125e-6; % Laser beam waist
lamL=532e-9; % laser wavelength
k=2*pi/lamL;
wL=k*c;
I=0.5e14;
t=D/v; % total interaction time

w2=2e-6; % second collimating slit width
l0=24e-2; % distance from incoherent source slit to collimating slit
l1=5.5e-2; % distance from second slit to grating
l2=24e-2; % distance from grating to screen

% V0=0;
V0=q^2*I/(2*m*eps0*c*wL^2); % Ponderomotive potential amplitude

dx01=10e-9;
x01=-w2/2:dx01:w2/2; % collimating second slit x-coordinate
x01=x01';

xmax1=5e-6;
dx1=10e-9;
x1=-xmax1:dx1:xmax1; % grating (laser) x-coordinate
x1=x1';

xmax2=1e-3;
dx2=1e-6;
x2=-xmax2:dx2:xmax2; % detection screen x-coordinate
x2=x2';

Vpond=V0*(cos(k*x1)).^2; % ponderomotive potential

w1=linspace(2e-6,20e-6,5);
cmax=zeros(size(w1)); % initializing central maximum position (shift)

for idx=1:length(w1)
    dx0=w1(idx)/10;
    x0=-w1(idx)/2:dx0:w1(idx)/2; % incoherent source slit x-coordinate
    x0=x0';
    
    P_detect=zeros(length(x2),length(x0)); % initializing the detection probability matrix

    % This section computes the propagation of the source wavefunction from the source to the one on the screen

    for iter=1:length(x0)
        Psi_inc=zeros(size(x0)); % initializign wavefunction at incoherent source slit
        Psi_slit=zeros(size(x01)); % initializign wavefunction at collimating slit
        Psi=zeros(size(x1)); % initializing wavefunction before laser interaction
        Psi_detect=zeros(size(x2)); % initializing wavefunction at detection screen
        
        % This section computes the source wavefunction (wf) using a incoherent slit
    
        Psi_inc(iter)=1;
    
        if iter==1 && idx==length(w1)%||iter==length(x0)
            figure
            subplot(2,1,1)
            stem(x0*1e6,Psi_inc,'.','linewidth',2)
            xlabel('Source position $x_0\ \mu m$','fontsize',15,'interpreter','latex')
            ylabel('$P_{inc}=\left|\Psi_{inc}(x_1)\right|^2$','fontsize',15,'interpreter','latex')
            axis([1.5*min(x0*1e6) 1.5*max(x0*1e6) min(Psi_inc) 1.2*max(Psi_inc)])
            grid on
        end
    
        % This section computes the wf propagation from the first (incoherent source) slit of width w1 to the second (collimating) slit of width w2

        Psi_slit=propagate(Psi_inc,x0,dx0,x01,l0);
        
        % This section computes the wf propagation from the second (collimating) slit to the laser

        Psi=propagate(Psi_slit,x01,dx01,x1,l1);

        % This section computes the wf after laser interaction
    
        Psi_grate=Psi.*exp(-1i*Vpond*t/hbar);

        % This section computes the wf propagation from the laser to the screen

        Psi_detect=propagate(Psi_grate,x1,dx1,x2,l2);
    
        P_detect(:,iter)=conj(Psi_detect).*Psi_detect; % This line computes the detection probability for each delta
        P_detect(:,iter)=P_detect(:,iter)/trapz(x2,P_detect(:,iter));
    
        if iter==1 && idx==length(w1)%||iter==length(x0)
            subplot(2,1,2)
            plot(x2*1e6,P_detect(:,iter))
            xlabel('Screen position $x_2\ \mu m$','fontsize',15,'interpreter','latex')
            ylabel('$P_{detect}=\left|\Psi_{detect}(x_2)\right|^2$','fontsize',15,'interpreter','latex')
            axis([min(x2*1e6) max(x2*1e6) min(P_detect(:,iter)) 1.2*max(P_detect(:,iter))])
            grid on
        end
    end

    cmax(idx)=x2(P_detect(:,1)==max(P_detect(:,1))); % storing the central maximum (due to first delta) shift
    
    % This section sums over the incoherent source

    P_final=sum(P_detect,2)*dx0;

    if idx==length(w1)
        figure
        plot(x2*1e6,P_final)
        xlabel('Screen position $x_2\ \mu m$','fontsize',15,'interpreter','latex')
        ylabel('$P_{final}=\left|\Psi_{final}(x_2)\right|^2$','fontsize',15,'interpreter','latex')
        axis([min(x2*1e6) max(x2*1e6) min(P_final) 1.2*max(P_final)])
        grid on
    end
end

figure
plot(w1*1e6/2,cmax*1e6)
grid on
xlabel('First delta location $\frac{w_1}{2}\ \mu m$','fontsize',15,'interpreter','latex')
ylabel('Central peak shift at detection screen $x_p\ \mu m$','fontsize',15,'interpreter','latex')

timeElapsed=string(toc) + ' seconds' % execution time in minutes

function Psi_out=propagate(Psi_in,x_in,dx,x_out,l)
    global lamdB;
	Psi_out=zeros(size(x_out));
	for i=1:length(x_out)
		for j=1:length(x_in)
			Kij=exp(1j*2*pi*sqrt((x_out(i)-x_in(j))^2+l^2)/lamdB);
			Psi_out(i)=Psi_out(i)+Kij*Psi_in(j)*dx;
        end
    end
end
