"""Ready-made configs for the MCP servers of services built for agents doing biology: Paperclip's
literature, Adaptyv's lab, and Proto's tools for protein and sequence design.

A preset is a server's entry in a config, written out already, with what its tools need a
person's approval for and what the agents are told of using them. A config names one with
preset, and anything else the entry gives is used in place of the preset's:

    mcp_servers:
      lab: {preset: adaptyv_testing}
      papers: {preset: paperclip, auth: oauth}     # signed in to, in place of an API key

or connect_mcp(presets=["paperclip", "proto"]) connects to presets by their own names, and
connect_mcp(presets={"paperclip": {"auth": "oauth"}}) with changes to them.
"""

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

# The tools of Adaptyv's server that only read, which are every one whose name begins list_ or
# get_, and the estimate of an experiment's cost. Every other tool, whatever its name, creates,
# changes, submits, pays for, or deletes something, so it waits for approval, which also holds
# for a tool the server adds later
ADAPTYV_READING = ("list_*", "get_*", "whoami", "health*", "cost_estimate")

ADAPTYV_INSTRUCTIONS = """\
Adaptyv's Foundry is a protein testing lab run as a service: it makes the proteins whose
sequences it is sent and measures them, in experiments of these types: screening and affinity
(binding to a target from its catalog, by BLI or SPR), thermostability, expression, and
fluorescence. An experiment is created as a draft; its cost can be estimated, in US cents, with
cost_estimate; it is then submitted, quoted, and the quote confirmed, which creates an invoice to
pay. Results arrive days to weeks later.

Every call that creates, changes, submits, confirms, pays for, or deletes something waits for a
person to approve it, and fails if they do not. Listing, getting, and estimating do not wait. So
before asking to create or submit an experiment, estimate its cost and say in the discussion what
it tests, with which sequences and target, and why. If a call is declined, do not make it again
unchanged: say what was declined, and go on without it."""

PAPERCLIP_INSTRUCTIONS = """\
Paperclip searches and reads the full text of over 11 million papers (bioRxiv, medRxiv, PubMed
Central, arXiv), over 150 million abstracts, regulatory documents from the FDA, EMA, and PMDA,
clinical trial registries, patents, and biological databases. Every paper it finds is citable by
its identifiers, such as its PMID or DOI: cite what you rely on by them. Searching and reading are
cheap; reading many papers at once with an AI reader is limited to 100 a day for the account, so
keep it for questions a search cannot answer."""

PROTO_INSTRUCTIONS = """\
Proto runs over a hundred bioinformatics tools, among them structure prediction, inverse folding,
protein and DNA language models, binder design, docking, and database lookups, on the user's own
Modal account, which is billed for what they run. Find a tool with search_tools or list_tools,
read its schema with get_tool_schema, and see a valid input with get_tool_example before the first
call to run_tool. Run small batches before large ones. Large outputs, such as structures, are
written to files on the machine running the server, and returned as their paths.

Deploying a tool that is not yet deployed builds and runs it on Modal at the user's expense, so a
call to deploy_tool waits for a person to approve it, and Proto's server then asks them to confirm
as well. If either is declined, nothing is deployed: use a deployed tool, or go on without it."""

# Proto's tools that read, or run a tool already deployed. Any other, such as deploy_tool, waits
# for approval
PROTO_ALLOWED = (
    "workspace_info",
    "list_tools",
    "search_tools",
    "get_tool_schema",
    "get_tool_example",
    "get_tool_info",
    "run_tool",
    "get_deploy_status",
    "list_runs",
    "get_run_status",
    "get_asset",
)


PROTO_HOSTED_INSTRUCTIONS = """\
Proto's hosted server runs bioinformatics tools, among them structure prediction, inverse
folding, protein and DNA language models, binder design, and database lookups, with nothing
installed here, under the Proto account signed in to, which may be charged for what is run.
Before the first call to a tool, find it and read what it takes with the server's tools for
doing so, and see a valid input where one is given. Run small batches before large ones.

Some calls, such as any that deploys a tool, wait for a person to approve them, and fail if they
do not. If a call is declined, do not make it again unchanged: say what was declined, and go on
without it."""


@dataclass(frozen=True)
class MCPPreset:
    """A ready-made config for one MCP server.

    :param entry: The server's entry in a config.
    :param about: What the server is, in a few words.
    :param setup: What to do before connecting, said when it cannot be connected to.
    :param credential: The header that carries the server's API key, which is left out where
        the server is signed in to instead, with auth: oauth; or None.
    :param signing_in: What to know of signing in to the server, said in place of setup where it
        is signed in to; or None.
    """

    entry: dict[str, Any] = field(repr=False)
    about: str
    setup: str = field(repr=False)
    credential: str | None = field(default=None, repr=False)
    signing_in: str | None = field(default=None, repr=False)

    def setup_for(self, entry: dict[str, Any]) -> str:
        """What to do before connecting to the server as an entry using the preset describes it."""
        if entry.get("auth") == "oauth" and self.signing_in is not None:
            return self.signing_in
        return self.setup


def signing_in(service: str, account: str) -> str:
    return (
        f"{service} is signed in to with {account}: the first connection opens its sign-in page in your browser, "
        "and the sign-in is kept for the next, in ~/.virtual_lab/mcp_auth unless VIRTUAL_LAB_MCP_AUTH_DIR says "
        f'elsewhere. sign_out_mcp("<preset or URL>") forgets it.'
    )


ADAPTYV_APPROVAL = {"ask": ["*"], "allow": list(ADAPTYV_READING)}

MCP_PRESETS: MappingProxyType[str, MCPPreset] = MappingProxyType(
    {
        "paperclip": MCPPreset(
            entry={
                "url": "https://paperclip.gxl.ai/mcp",
                "headers": {"X-API-Key": "${PAPERCLIP_API_KEY}"},
                "instructions": PAPERCLIP_INSTRUCTIONS,
            },
            about="Paperclip: search and read papers, trials, patents, and regulatory documents",
            setup=(
                "Paperclip needs an API key, which starts gxl_, in PAPERCLIP_API_KEY, in the environment or the .env "
                "file. Make one at https://paperclip.gxl.ai/keys, or sign in instead, with auth: oauth in the "
                "server's entry."
            ),
            credential="X-API-Key",
            signing_in=signing_in("Paperclip", "your Paperclip account"),
        ),
        "adaptyv": MCPPreset(
            # The trailing slash is part of the address the server knows itself by
            entry={
                "url": "https://mcp.adaptyvbio.com/mcp/",
                "headers": {"Authorization": "Bearer ${FOUNDRY_API_TOKEN}"},
                "approval": ADAPTYV_APPROVAL,
                "instructions": ADAPTYV_INSTRUCTIONS,
            },
            about="Adaptyv Foundry: have proteins made and tested in a lab, which costs money",
            setup=(
                "Adaptyv's Foundry needs an API token in FOUNDRY_API_TOKEN, in the environment or the .env file. Make "
                "one in the Foundry portal, https://foundry.adaptyvbio.com, under Organization, Settings, Tokens. A "
                "read-only one, made with /tokens/attenuate, is enough to list and read experiments. Or sign in "
                "instead, with auth: oauth in the server's entry. To try it without ordering anything, use the preset "
                "adaptyv_testing."
            ),
            credential="Authorization",
            signing_in=signing_in("Adaptyv's Foundry", "your Foundry account"),
        ),
        "adaptyv_testing": MCPPreset(
            entry={
                "url": "https://mcp.testing.adaptyvbio.com/mcp/",
                "headers": {"Authorization": "Bearer ${FOUNDRY_TESTING_TOKEN}"},
                "approval": ADAPTYV_APPROVAL,
                "instructions": (
                    ADAPTYV_INSTRUCTIONS
                    + "\n\nThis is Adaptyv's testing sandbox, where nothing is made or charged for: it is for trying "
                    "calls out."
                ),
            },
            about="Adaptyv Foundry's testing sandbox, where nothing is ordered or charged",
            setup=(
                "Adaptyv's testing sandbox needs a token of its own, in FOUNDRY_TESTING_TOKEN, in the environment or "
                "the .env file. Make an account and a token at https://foundry.testing.adaptyvbio.com, or sign in "
                "instead, with auth: oauth in the server's entry."
            ),
            credential="Authorization",
            signing_in=signing_in("Adaptyv's testing sandbox", "an account of the sandbox"),
        ),
        "proto": MCPPreset(
            entry={
                "command": "proto-tools-mcp",
                "approval": {"ask": ["*"], "allow": list(PROTO_ALLOWED)},
                "instructions": PROTO_INSTRUCTIONS,
            },
            about="Proto: run tools for protein and sequence design on your own Modal account",
            setup=(
                "Proto's server runs here, with proto-tools-mcp, and runs tools on your Modal account. Install it with "
                'uv tool install "proto-tools[mcp] @ git+https://github.com/evo-design/proto-tools.git", sign in to '
                "Modal with modal token new, and deploy the tools you want with proto-tools deploy --apps <name>. Or "
                "use the preset proto_hosted, which needs nothing installed."
            ),
        ),
        "proto_hosted": MCPPreset(
            entry={
                "url": "https://mcp.evodesign.org/mcp",
                "auth": "oauth",
                "approval": {"ask": ["*"], "allow": list(PROTO_ALLOWED)},
                "instructions": PROTO_HOSTED_INSTRUCTIONS,
            },
            about="Proto's hosted server: its tools with nothing installed here, signed in to with a Proto account",
            setup=signing_in("Proto's hosted server", "a Proto account, from https://proto.evodesign.org"),
        ),
    }
)


def preset_entry(server: str, entry: dict[str, Any]) -> tuple[dict[str, Any], MCPPreset | None]:
    """A server's entry with the preset it names filled in, and the preset, or None if it names
    none."""
    if "preset" not in entry:
        return entry, None
    name = entry["preset"]
    if not isinstance(name, str) or name not in MCP_PRESETS:
        listed = "; ".join(f"{key} ({preset.about})" for key, preset in MCP_PRESETS.items())
        raise ValueError(f"The MCP server {server} uses the preset {name!r}, which is not one. The presets are: {listed}")
    preset = MCP_PRESETS[name]

    # A copy, so that nothing done with the entry changes the preset
    filled = copy.deepcopy(preset.entry)
    given = {key: value for key, value in entry.items() if key != "preset"}
    # A server signed in to is not sent the API key it would otherwise be, nor needs one set
    if given.get("auth") == "oauth" and preset.credential is not None and "headers" in filled:
        filled["headers"] = {key: value for key, value in filled["headers"].items() if key != preset.credential}
        if not filled["headers"]:
            del filled["headers"]

    return {**filled, **given}, preset
