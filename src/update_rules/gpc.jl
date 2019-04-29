@naiveVariationalRule(:node_type       => GPC,
                      :outbound_type   => Message{Gaussian},
                      :inbound_types   =>(Nothing,ProbabilityDistribution,ProbabilityDistribution),
                      :name            =>VBGPCOutVPP)

@structuredVariationalRule(:node_type     => GPC,
                           :outbound_type => Message{Gaussian},
                           :inbound_types => (Nothing, Message{Gaussian}, ProbabilityDistribution),
                           :name          => SVBGPCOutVGG)

@structuredVariationalRule(:node_type     => GPC,
                           :outbound_type => Message{Gaussian},
                           :inbound_types => (Message{Gaussian}, Nothing, ProbabilityDistribution),
                           :name          => SVBGPCMeanGVG)

@structuredVariationalRule(:node_type     => GPC,
                           :outbound_type => Message{Gaussian},
                           :inbound_types => (ProbabilityDistribution, Nothing),
                           :name          => SVBGPCVarGV)

@marginalRule(:node_type => GPC,
             :inbound_types => (Message{Gaussian}, Message{Gaussian}, ProbabilityDistribution),
             :name => MGPCGGD)
