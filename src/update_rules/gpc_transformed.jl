@naiveVariationalRule(:node_type       => GPCLinear,
                      :outbound_type   => Message{Gaussian},
                      :inbound_types   =>(Nothing,ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
                      :name            =>VBGPCLinearOutVPP)

@structuredVariationalRule(:node_type     => GPCLinear,
                           :outbound_type => Message{Gaussian},
                           :inbound_types => (Nothing, Message{Gaussian}, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
                           :name          => SVBGPCLinearOutVGGPP)

@structuredVariationalRule(:node_type     => GPCLinear,
                           :outbound_type => Message{Gaussian},
                           :inbound_types => (Message{Gaussian}, Nothing, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
                           :name          => SVBGPCLinearMeanGVGPP)

@structuredVariationalRule(:node_type     => GPCLinear,
                           :outbound_type => Message{Gaussian},
                           :inbound_types => (ProbabilityDistribution, Nothing,ProbabilityDistribution,ProbabilityDistribution),
                           :name          => SVBGPCLinearVarGVPP)

@marginalRule(:node_type => GPCLinear,
             :inbound_types => (Message{Gaussian}, Message{Gaussian}, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
             :name => MGPCLinearGGDDD)
