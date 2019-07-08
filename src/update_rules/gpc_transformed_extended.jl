@naiveVariationalRule(:node_type    => GPCLinearExtended,
                      :outbound_type=> Message{Gaussian},
                      :inbound_types=> (Nothing, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
                      :name         => VBGPCLinearExtendedOutVGGPP)
@naiveVariationalRule(:node_type    => GPCLinearExtended,
                      :outbound_type=> Message{Gaussian},
                      :inbound_types=> (ProbabilityDistribution,Nothing,ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
                      :name         => VBGPCLinearExtendedMeanGVGPP)
@naiveVariationalRule(:node_type    => GPCLinearExtended,
                      :outbound_type=> Message{Gaussian},
                      :inbound_types=> (ProbabilityDistribution,ProbabilityDistribution,Nothing,ProbabilityDistribution,ProbabilityDistribution),
                      :name         => VBGPCLinearExtendedVarGGVPP)

@structuredVariationalRule(:node_type     => GPCLinearExtended,
                         :outbound_type => Message{Gaussian},
                         :inbound_types => (Nothing, Message{Gaussian}, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
                         :name          => SVBGPCLinearExtendedOutVGGPP)

@structuredVariationalRule(:node_type     => GPCLinearExtended,
                         :outbound_type => Message{Gaussian},
                         :inbound_types => (Message{Gaussian}, Nothing, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
                         :name          => SVBGPCLinearExtendedMeanGVGPP)

@structuredVariationalRule(:node_type     => GPCLinearExtended,
                         :outbound_type => Message{Gaussian},
                         :inbound_types => (ProbabilityDistribution, Nothing,ProbabilityDistribution,ProbabilityDistribution),
                         :name          => SVBGPCLinearExtendedVarGVPP)

@marginalRule(:node_type => GPCLinearExtended,
           :inbound_types => (Message{Gaussian}, Message{Gaussian}, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
           :name => MGPCLinearExtendedGGDDD)
