# # INTENTION

This specification establishes the foundational operational framework governing AI agent activities within the SpecRegistry system. It details mandatory procedures for evaluating, incorporating, and exhibiting responsible behavior toward specified guidelines vital for project integrity, governance, and transparency. Ensuring adherence to these rules facilitates predictable operation and enhances overall system dependability.

## # SCOPE

The scope encompasses all autonomous intelligence tools, agents, and scripting engines that leverage the SpecRegistration nexus—specifically those utilizing MCP server deployments. It extends from basic operator roles to sophisticated integration points requiring stringent adherence to documented requirements.

## # PURPOSE

Implementing these explicit operational protocols directly enhances system reliability by enforcing consistent contextualisation, proactively managing model judgment processes, and actively promoting accurate conformance to governed specifications. This streamlined methodology fosters robust system stability relative to the threat of unstated biases or inconsistencies. 

## # REQUIREMENTS

1.  **MCP Deployment:** Require agents to consistently utilize the MCP server as a primary access point for all core operations, prioritizing `get_specs` upon instantiation.
2.  **Specification Search Paradigm:** Mandate utilization from structured searching in SpecRegistry’s domain-specific search index (`search_specs`) when working with large context parameter sets. 
3. **Guidance Processing Hierarchy**: Prior to engaging with comprehensive, locally conceived specifications, each agent must call `resolve_guidance` or the proposed Agent API function to obtain proper style guides and technical documentation – this needs to occur when any coverage is not yet documented—to avoid introducing inconsistencies, enabling a more robust mechanism for resource-based governance.
4.  **Repo Configuration:** Adherence to a SPECREG\_REPO variable in all repository settings ensures effective project separation, bolstering autonomy within specific sections between core code files and system context. 
5   **Auth/Token Management:** Enforce strict handling of `SPECRED_TOKEN` via authentication using the dedicated CLI, preventing exposure. Access requirements must prevent unauthorized or misusing parameters from containing sensitive code. 
6.  **Specification Citation Protocol**: All cited specs must show their presence through the specific identifier in details such as the references section and are accessible as standard.
7.   **Feedback & Reporting:** Agents explicitly document all feedback, deviations, discrepancies, or areas requiring resolution in direct output formats—specifically designated in requirements within the SpecRegistry system – for proactive data analysis and configuration-optimized operational standards. 
8. **Specification Isolation**:Agents shall not traverse the system explorer or examination of the registry; each agent must engage with core OS methods or a dedicated API to uphold these rules regarding operating.