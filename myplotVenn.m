%Plots probability and accuracy 
%      function myplot(low,up,a)
function myplotVenn(low,up,a)
linewidth = 1;
fontsize = 12;
%set(gca,'XTick',[1:1:length(low)] );
axes('FontSize',fontsize);
plot(low,':k','LineWidth',linewidth);
hold on;
plot(up,'b','LineWidth',linewidth);
hold on;
%plot((low+up)/2,'-.b','LineWidth',linewidth);
plot(a,'--r','LineWidth',linewidth);
%axis([0 130000 0.5 1]);
%set(gca,'XLim',[100,1800]);
%set(gca,'XTick',[1:length(low)]);
%set(gca,'XTickLabel',[100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,3500,4000,4600]);
set(gca,'XTickLabel',[0:100:800]);
xlabel('example #','fontsize',fontsize);
%ylabel('Percentage','fontsize',fontsize);
legend('lower bound','upper bound','accuracy','Location','SouthEast');
end

