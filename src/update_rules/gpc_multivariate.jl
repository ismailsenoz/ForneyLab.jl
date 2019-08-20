@structuredVariationalRule(:node_type => MultivariateGPC,
                           :outbound_type => Message{Gaussian},
                           :inbound_types => (Nothing, Message{Gaussian}, ProbabilityDistribution),
                           :name => SVBMultivariateGPCOutNGD)

@structuredVariationalRule(:node_type => MultivariateGPC,
                           :outbound_type => Message{Gaussian},
                           :inbound_types => (Message{Gaussian}, Nothing, ProbabilityDistribution),
                           :name => SVBMultivariateGPCMeanGND)

@structuredVariationalRule(:node_type => MultivariateGPC,
                           :outbound_type => Message{LogDetTrace},
                           :inbound_types => (ProbabilityDistribution, Nothing),
                           :name => SVBMultivariateGPCCovDN)

@marginalRule(:node_type => MultivariateGPC,
              :inbound_types => (Message{Gaussian}, Message{Gaussian}, ProbabilityDistribution),
              :name => MMultivariateGPCGGD)
