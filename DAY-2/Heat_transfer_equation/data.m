initialAndBoundaryConditions = readtable("Condition.xlsx",VariableNamingRule="preserve");
thermalmodel = createpde("thermal","transient");
geometryFromEdges(thermalmodel,@crackg);
thermalProperties(thermalmodel,...
    ThermalConductivity=1,...
    MassDensity=1,...
    SpecificHeat=1);
generateMesh(thermalmodel,GeometricOrder="linear");

pdegplot(thermalmodel,"EdgeLabels","on");
solutionTimes = 0:0.01:0.1;

numObservations = height(initialAndBoundaryConditions);

initialCondition = initialAndBoundaryConditions.("Initial Condition");
dirichletBoundary = initialAndBoundaryConditions.E6_Temperature;
neumannBoundary = initialAndBoundaryConditions.E1_HeatFlux;

parfor i = 1:numObservations-1    
    thermalIC(thermalmodel,initialCondition(i)/100);
    fprintf("Dirich oservation: %d, and value %f\n", i, dirichletBoundary(i)/100);
    thermalBC(thermalmodel, Edge=6, Temperature = dirichletBoundary(i)/100);
    thermalBC(thermalmodel,Edge = 1, HeatFlux = neumannBoundary(i)/100);
    solution{i} = solve(thermalmodel,solutionTimes);
end
save('data.mat', 'solution', '-v7.3')

figure
pdeplot(thermalmodel,XYData=solution{1}.Temperature(:,end),Contour="on",ColorMap="jet");


% Choose only the mesh nodes corresponding to interior points
x = thermalmodel.Mesh.Nodes.';
numNodes = size(x,1);
numEdges = thermalmodel.Geometry.NumEdges;
edgeNodeId = thermalmodel.Mesh.findNodes("region","Edge",1:numEdges);
interiorNodeId = 1:numNodes;
interiorNodeId(edgeNodeId) = [];
interiorNodes = thermalmodel.Mesh.Nodes(:,interiorNodeId);
