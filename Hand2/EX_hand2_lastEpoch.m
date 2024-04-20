clear; tic
% close all
load hand2.txt
load hand2_int.txt
dsites=hand2;

% figure (1)
% plot3(x,y,z,'ro',dsites(:,1),dsites(:,2),dsites(:,3),'b.','MarkerFaceColor',[1 0 0]);hold on
% xlabel('x')
% view([0,90]); axis equal; 

neval=70; 
% %%%%%%%%%% Evaluation points
bmin=min(dsites,[],1);  bmax=max(dsites,[],1);
xgrid=linspace(bmin(1),bmax(1)+.1,neval);
ygrid=linspace(bmin(2),bmax(2),neval);
zgrid=linspace(bmin(3),bmax(3),neval);
[xe,ye,ze]=meshgrid(xgrid,ygrid,zgrid);
epoints=[xe(:),ye(:),ze(:)]; %clear xgrid ygrid zgrid
%%
save('griddata.mat','epoints')
%%
load('pf_data.mat')  
% Access the loaded variables
pf = pfa;  % Access the 'pf1' variable
% pf_list = pfa_list';
%%
figure ()
pfit=patch(isosurface(xe,ye,ze,reshape(pf,neval,neval,neval),0));
% patch(isocaps(xe,ye,ze,reshape(pf,neval,neval,neval),0,'above'),'FaceColor','interp','EdgeColor','none')
isonormals(xe,ye,ze,reshape(pf,neval,neval,neval),pfit)
set(pfit,'FaceLighting','gouraud','FaceColor',[0.93,0.69,0.13],'EdgeColor','none');
set(pfit,'FaceLighting','gouraud','FaceColor','g','EdgeColor','none');
light('Position',[ 1 0 1],'Style','infinite');
light('Position',[  0.2 0.2 -.5],'Style','local'); %hand lighting
light('Position',[  -1 0.2 .1],'Style','local');
daspect([1 1 1]);  
% colormap jet
% colorbar
view([120,120]); axis off
%  title('Epoch 1800')
set(gca,'FontSize',16)
toc